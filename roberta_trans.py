from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import pandas as pd
import torch
import numpy as np

def prepare_multilabel_roberta_dataset(X_train, y_train, X_val, y_val, X_test, y_test, model_id="cardiffnlp/twitter-xlm-roberta-base", max_length=128):
    """
    Prepara datasets de train, val y test para multietiqueta con RoBERTa.

    Args:
        X_train, y_train, X_val, y_val, X_test, y_test: Series o listas. y debe tener varias columnas (una por etiqueta)
        model_id: modelo de HuggingFace
        max_length: longitud máxima de tokens
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def create_dataset(X, y):
        df = pd.DataFrame(X)
        df["text"] = X
        labels = pd.DataFrame(y)
        df = pd.concat([df, labels], axis=1).dropna()

        dataset = Dataset.from_pandas(df)

        def tokenize(batch):
            encodings = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length)
            # Convierte todas las columnas de etiquetas en un array numpy
            encodings["labels"] = np.array(batch[y.columns].to_list())
            return encodings

        dataset = dataset.map(tokenize, batched=True)
        dataset.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])
        return dataset

    train_dataset = create_dataset(X_train, y_train)
    val_dataset = create_dataset(X_val, y_val)
    test_dataset = create_dataset(X_test, y_test)

    num_labels = y_train.shape[1]

    return tokenizer, train_dataset, val_dataset, test_dataset, num_labels

def train_multilabel_roberta(train_dataset, val_dataset, num_labels, model_id="cardiffnlp/twitter-xlm-roberta-base", output_dir="./roberta_multilabel", epochs=2, batch_size=2, lr=2e-5):
    """
    Entrena RoBERTa en un problema multietiqueta.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=num_labels,
        problem_type="multi_label_classification"
    )

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

    outputs = trainer.train()
    return trainer, outputs
