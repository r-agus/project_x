import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

from TextVectorRepresentation import load_data, divide_train_val_test, separate_x_y_vectors
from TextVectorRepresentation import vectorRepresentation_TFIDF
from TextVectorRepresentation import vectorRepresentation_Word2Vec
from TextVectorRepresentation import vectorRepresentation_BERT

DATA_PATH = "Datasets/EvaluationData/politicES_phase_2_train_public.csv"
TEXT_COL = "tweet"
TARGET_COL = "ideology_binary"  

# To encode string labels into numbers
def encode_labels(y_train, y_val, y_test):
    """
    Encodes string labels into integer values for machine learning models.

    This function fits a LabelEncoder on the training labels and applies the same
    encoding to the validation and test labels, ensuring consistency across splits.

    Parameters
    ----------
    y_train : array-like
        Target labels from the training set.
    y_val : array-like
        Target labels from the validation set.
    y_test : array-like
        Target labels from the test set.

    Returns
    -------
    y_train_enc : ndarray
        Encoded labels for the training set.
    y_val_enc : ndarray
        Encoded labels for the validation set.
    y_test_enc : ndarray
        Encoded labels for the test set.
    encoder : LabelEncoder
        The fitted LabelEncoder instance, useful for inverse-transforming predictions.
    """
    encoder = LabelEncoder()
    y_train_enc = encoder.fit_transform(y_train)
    y_val_enc = encoder.transform(y_val)
    y_test_enc = encoder.transform(y_test)
    return y_train_enc, y_val_enc, y_test_enc, encoder

# Build Word2Vec embeddings
def build_word2vec(X_train, X_val, X_test):
    """
    Generates Word2Vec embeddings for train, validation and test sets
    using your group's implementation.
    """

    train_w2v, val_w2v, test_w2v = vectorRepresentation_Word2Vec(
        X_train, X_val, X_test
    )

    return train_w2v, val_w2v, test_w2v

# Build BERT embeddings
def build_bert(X_train, X_val, X_test):
    """
    Generates BERT embeddings for train, validation and test sets
    using your group's implementation.
    """
    train_bert, val_bert, test_bert = vectorRepresentation_BERT(
        X_train, X_val, X_test
    )
    return train_bert, val_bert, test_bert




def train_and_evaluate_model(model_type, X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Trains and evaluates a machine learning model (Logistic Regression or Linear SVM)
    using the provided feature representations (TF-IDF, Word2Vec or BERT).

    The function performs:
        1. Model creation based on `model_type`
        2. Training on the training set
        3. Evaluation on validation and test sets
        4. Printing of accuracy, macro F1-score, classification report,
           and confusion matrix for both validation and test sets.

    Parameters
    ----------
    model_type : str
        Type of model to train. Must be:
            - "logreg" → Logistic Regression
            - "svm" → Linear SVM
    X_train : array-like or sparse matrix
        Feature vectors for the training set.
    y_train : array-like
        Encoded labels for the training set.
    X_val : array-like or sparse matrix
        Feature vectors for the validation set.
    y_val : array-like
        Encoded labels for the validation set.
    X_test : array-like or sparse matrix
        Feature vectors for the test set.
    y_test : array-like
        Encoded labels for the test set.

    Returns
    -------
    model : estimator object
        The trained model instance.
    accuracy : float
        Accuracy score on the test set.
    f1_macro : float
        Macro-averaged F1 score on the test set.
    """

    # We create the model
    if model_type == "logreg":
        print("\nTraining Logistic Regression...")
        model = LogisticRegression(max_iter=2000, n_jobs=-1, class_weight="balanced")

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
    Runs a model experiment using three text representations: TF-IDF, Word2Vec and BERT.
    Trains and evaluates the specified model type on each representation.

    Returns
    -------
    dict
        Dictionary with accuracy and macro F1-score for each representation:
        {
            "tfidf": (accuracy, f1_macro),
            "word2vec": (accuracy, f1_macro),
            "bert": (accuracy, f1_macro)
        }
    """
    print(f"\n{model_type.upper()} EXPERIMENT: {exp_name}")

    # Load dataset
    data = load_data(DATA_PATH)

    # Divide into train, validation, test
    train_df, val_df, test_df = divide_train_val_test(data)

    # Separate X and y
    X_train, y_train_df = separate_x_y_vectors(train_df)
    X_val, y_val_df = separate_x_y_vectors(val_df)
    X_test, y_test_df = separate_x_y_vectors(test_df)

    # Extract target column (ideology_binary or ideology_multiclass)
    y_train = y_train_df[target_col]
    y_val = y_val_df[target_col]
    y_test = y_test_df[target_col]

    # Labels -> numbers (integers)
    y_train_enc, y_val_enc, y_test_enc, encoder = encode_labels(y_train, y_val, y_test)

    # TF-IDF vectorization
    X_train_tfidf, X_val_tfidf, X_test_tfidf = vectorRepresentation_TFIDF(X_train, X_val, X_test)

    # WORD2VEC representation
    X_train_w2v, X_val_w2v, X_test_w2v = build_word2vec(X_train, X_val, X_test)

    # BERT representation
    X_train_bert, X_val_bert, X_test_bert = build_bert(X_train, X_val, X_test)


    results = {}

    # TF-IDF
    print("\n USING TF-IDF REPRESENTATION ")
    model_tfidf, acc_tfidf, f1_tfidf = train_and_evaluate_model(
        model_type,
        X_train_tfidf, y_train_enc,
        X_val_tfidf, y_val_enc,
        X_test_tfidf, y_test_enc
    )
    results["tfidf"] = (acc_tfidf, f1_tfidf)

    # WORD2VEC
    print("\n USING WORD2VEC REPRESENTATION ")
    model_w2v, acc_w2v, f1_w2v = train_and_evaluate_model(
        model_type,
        X_train_w2v, y_train_enc,
        X_val_w2v, y_val_enc,
        X_test_w2v, y_test_enc
    )
    results["word2vec"] = (acc_w2v, f1_w2v)

    # BERT 
    print("\n USING BERT REPRESENTATION ")
    model_bert, acc_bert, f1_bert = train_and_evaluate_model(
        model_type,
        X_train_bert, y_train_enc,
        X_val_bert, y_val_enc,
        X_test_bert, y_test_enc
    )
    results["bert"] = (acc_bert, f1_bert)


    return results

results = []
if __name__ == "__main__":

    # LOGISTIC REGRESSION - BINARY
    res = run_model_experiment("logreg", "ideology_binary", "Binary classification")

    results.append(["LogReg + TF-IDF", "Binary", res["tfidf"][0], res["tfidf"][1]])
    results.append(["LogReg + Word2Vec", "Binary", res["word2vec"][0], res["word2vec"][1]])
    results.append(["LogReg + BERT", "Binary", res["bert"][0], res["bert"][1]])

    # LOGISTIC REGRESSION - MULTICLASS
    res = run_model_experiment("logreg", "ideology_multiclass", "Multiclass classification")

    results.append(["LogReg + TF-IDF", "Multiclass", res["tfidf"][0], res["tfidf"][1]])
    results.append(["LogReg + Word2Vec", "Multiclass", res["word2vec"][0], res["word2vec"][1]])
    results.append(["LogReg + BERT", "Multiclass", res["bert"][0], res["bert"][1]])

    # SVM - BINARY
    res = run_model_experiment("svm", "ideology_binary", "Binary classification")

    results.append(["SVM + TF-IDF", "Binary", res["tfidf"][0], res["tfidf"][1]])
    results.append(["SVM + Word2Vec", "Binary", res["word2vec"][0], res["word2vec"][1]])
    results.append(["SVM + BERT", "Binary", res["bert"][0], res["bert"][1]])

    # SVM - MULTICLASS
    res = run_model_experiment("svm", "ideology_multiclass", "Multiclass classification")

    results.append(["SVM + TF-IDF", "Multiclass", res["tfidf"][0], res["tfidf"][1]])
    results.append(["SVM + Word2Vec", "Multiclass", res["word2vec"][0], res["word2vec"][1]])
    results.append(["SVM + BERT", "Multiclass", res["bert"][0], res["bert"][1]])

    # PRINT TABLE
    print("\n\n================== COMPARATIVE TABLE ===================")
    print("{:<22} {:<12} {:<10} {:<10}".format("Model", "Task", "Accuracy", "F1-macro"))
    print("--------------------------------------------------------")
    for row in results:
        print("{:<22} {:<12} {:.4f}     {:.4f}".format(row[0], row[1], row[2], row[3]))