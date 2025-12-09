from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import pandas as pd
import torch

def prepare_roberta_dataset(X_train, y_train, X_val, y_val, X_test, y_test, model_id="cardiffnlp/twitter-xlm-roberta-base", max_length=128):
    """
    Prepares the train, validation, and test datasets for RoBERTa using HuggingFace datasets.

    Args:
        X_train, y_train, X_val, y_val, X_test, y_test: Datasets as lists, pandas Series, or numpy arrays.
        model_id (str): HuggingFace model ID.
        max_length (int): Maximum token length for tokenization.

    Returns:
        tokenizer: Tokenizer object.
        train_dataset, val_dataset, test_dataset: Torch datasets ready for Trainer.
        num_labels (int): Number of unique labels.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def create_dataset(X, y):
        df = pd.DataFrame({
            "tweet": X,
            "labels": y
        }).dropna()

        # Convert object labels to numeric if needed
        if df["labels"].dtype == "object":
            df["labels"] = df["labels"].astype("category").cat.codes

        dataset = Dataset.from_pandas(df)

        def tokenize(batch):
            return tokenizer(batch["tweet"], truncation=True, padding="max_length", max_length=max_length)

        dataset = dataset.map(tokenize, batched=True)
        dataset.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])
        return dataset, df["labels"].nunique()

    train_dataset, num_labels_train = create_dataset(X_train, y_train)
    val_dataset, num_labels_val = create_dataset(X_val, y_val)
    test_dataset, num_labels_test = create_dataset(X_test, y_test)

    # Ensure number of labels is consistent
    assert num_labels_train == num_labels_val == num_labels_test, "Mismatch in number of labels between datasets"

    return tokenizer, train_dataset, val_dataset, test_dataset, num_labels_train

def train_roberta(train_dataset, val_dataset, num_labels, model_id="cardiffnlp/twitter-xlm-roberta-base", output_dir="./roberta_model", epochs=2, batch_size=2, lr=2e-5):
    """
    Trains a RoBERTa model using HuggingFace Trainer.

    Args:
        train_dataset, val_dataset: Torch datasets.
        num_labels (int): Number of labels for classification.
        model_id (str): HuggingFace model ID.
        output_dir (str): Directory to save model outputs.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size.
        lr (float): Learning rate.

    Returns:
        trainer: HuggingFace Trainer object after training.
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=num_labels)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        num_train_epochs=epochs,
        evaluation_strategy="epoch",
        logging_steps=20,
        fp16=torch.cuda.is_available()
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    trainer.train()
    return trainer


if __name__ == "__main__":
    # Asume que ya tienes X_train, y_train, X_val, y_val, X_test, y_test
    from init import xtrain, ytrain, xval, yval, xtest, ytest

    tokenizer, train_dataset, val_dataset, test_dataset, num_labels = prepare_roberta_dataset(
        xtrain, ytrain, xval, yval, xtest, ytest
    )

    trainer = train_roberta(train_dataset, val_dataset, num_labels)
