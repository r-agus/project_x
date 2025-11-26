import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

from stopwords import stopwords 

DATA_PATH = "Datasets/EvaluationData/politicES_phase_2_train_public.csv"
TEXT_COL = "tweet"
TARGET_COL = "ideology_binary"  

def load_and_split(data_path, text_col=TEXT_COL, target_col=TARGET_COL,
                   test_size=0.2, val_size=0.1, random_state=42):
    df = pd.read_csv(data_path)
    # We delete the rows with no text or no label
    df = df.dropna(subset=[text_col, target_col])

    X = df[text_col].astype(str)
    y = df[target_col]
    #Now we split the data into train, val and test 
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(test_size + val_size), stratify=y, random_state=random_state)

    # Now X_temp and y_temp contain both val and test data, we need to split them again to get val and test sets
    val_ratio = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_ratio), 
        stratify=y_temp,
        random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

# To encode string labels into numbers
def encode_labels(y_train, y_val, y_test):
    encoder = LabelEncoder()
    y_train_enc = encoder.fit_transform(y_train)
    y_val_enc = encoder.transform(y_val)
    y_test_enc = encoder.transform(y_test)
    return y_train_enc, y_val_enc, y_test_enc, encoder


# TF-IDF with our custom stopwords list
def build_tfidf(X_train, X_val, X_test):
    vectorizer = TfidfVectorizer(
        max_features=20_000,
        ngram_range=(1, 2),
        stop_words=list(stopwords)  # our custom stopwords list
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)

    return vectorizer, X_train_tfidf, X_val_tfidf, X_test_tfidf


def train_and_evaluate_model(model_type, X_train, y_train, X_val, y_val, X_test, y_test):

    # We create the model - Logistic Regression
    if model_type == "logreg":
        print("\nTraining Logistic Regression...")
        model = LogisticRegression(max_iter=2000, n_jobs=-1,class_weight="balanced")

    elif model_type == "svm":
        print("\nTraining Linear SVM...")
        model = LinearSVC(class_weight="balanced")
    else:
        raise ValueError("model_type must be 'logreg' or 'svm'")

    # We train the model on the training set

    # The model learns the relationship between TF-IDF features and the labels
    model.fit(X_train, y_train)

    # We make predictions on the VALIDATION set

    y_pred_val = model.predict(X_val)

    # Print evaluation metrics for validation set

    print(f"\n{model_type.upper()} - VALIDATION")
    # Accuracy
    print(f"Accuracy: {accuracy_score(y_val, y_pred_val):.4f}")

    # To give equal importance to both classes
    print(f"F1-score: {f1_score(y_val, y_pred_val, average='macro'):.4f}")

    # Classification report to know if the model is biased towards one class
    print("\nClassification Report (Validation):")
    print(classification_report(y_val, y_pred_val))

    # To know how many examples the model predicted correctly and where it is making mistakes
    print("Confusion Matrix (Validation):")
    print(confusion_matrix(y_val, y_pred_val))


    # We make predictions on the TEST set

    y_pred_test = model.predict(X_test)

    print(f"\n{model_type.upper()} - TEST")

    # Accuracy on test
    print(f"Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")

    # To give equal importance to both classes
    print(f"F1-score: {f1_score(y_test, y_pred_test, average='macro'):.4f}")

    # The classification report to know if the model is biased towards one class
    print("\nClassification Report (Test):")
    print(classification_report(y_test, y_pred_test))

    # Confusion matrix for the test set, so we know how many examples predicted
    print("Confusion Matrix (Test):")
    print(confusion_matrix(y_test, y_pred_test))

    # It returns the trained model and the metrics
    return model, accuracy_score(y_test, y_pred_test), f1_score(y_test, y_pred_test, average='macro')


def run_model_experiment(model_type, target_col, exp_name):
    """
    Runs a model experiment by training and evaluating a specified model type on the given target column.

    Parameters:
        model_type (str): The type of model to train ('logreg' for Logistic Regression, 'svm' for Support Vector Machine).
        target_col (str): The name of the target column in the dataset.
        exp_name (str): A descriptive name for the experiment.

    Returns:
        model: The trained model instance.
        encoder: The label encoder used for target labels.
        vectorizer: The TF-IDF vectorizer used for text features.
        acc (float): Accuracy score on the test set.
        f1 (float): Macro-averaged F1 score on the test set.
    """
    print(f"\n{model_type.upper()} EXPERIMENT: {exp_name}")

    # Load and split the data
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split(
        DATA_PATH,
        text_col=TEXT_COL,
        target_col=target_col
    )

    # Labels -> numbers (integers)
    y_train_enc, y_val_enc, y_test_enc, encoder = encode_labels(y_train, y_val, y_test)

    # TF-IDF vectorization
    vectorizer, X_train_tfidf, X_val_tfidf, X_test_tfidf = build_tfidf(
        X_train, X_val, X_test
    )

    # Train and evaluate Logistic Regression
    model, acc, f1 = train_and_evaluate_model(model_type, X_train_tfidf, y_train_enc, X_val_tfidf, y_val_enc, X_test_tfidf, y_test_enc)

    return model, encoder, vectorizer, acc, f1

results = []
if __name__ == "__main__":

    # LOGISTIC REGRESSION - BINARY
    model, encoder, vectorizer, acc, f1 = run_model_experiment(
        "logreg", "ideology_binary", "Binary classification"
    )
    results.append(["Logistic Regression", "Binary", acc, f1])

    # LOGISTIC REGRESSION - MULTICLASS
    model, encoder, vectorizer, acc, f1 = run_model_experiment(
        "logreg", "ideology_multiclass", "Multiclass classification"
    )
    results.append(["Logistic Regression", "Multiclass", acc, f1])

    # SVM - BINARY
    model, encoder, vectorizer, acc, f1 = run_model_experiment(
        "svm", "ideology_binary", "Binary classification"
    )
    results.append(["SVM", "Binary", acc, f1])

    # SVM - MULTICLASS
    model, encoder, vectorizer, acc, f1 = run_model_experiment(
        "svm", "ideology_multiclass", "Multiclass classification"
    )
    results.append(["SVM", "Multiclass", acc, f1])

    # ======= PRINT TABLE =======
    print("\n\n================== COMPARATIVE TABLE ===================")
    print("{:<22} {:<12} {:<10} {:<10}".format("Model", "Task", "Accuracy", "F1-macro"))
    print("--------------------------------------------------------")
    for row in results:
        print("{:<22} {:<12} {:.4f}     {:.4f}".format(row[0], row[1], row[2], row[3]))