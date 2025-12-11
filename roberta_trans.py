from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os

# ----------------------------
# Función de métricas
# ----------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# ----------------------------
# Preparación de datasets para varias etiquetas
# ----------------------------
def prepare_multilabel_datasets(
    X_train, y_train, X_val, y_val, X_test, y_test,
    label_list,
    model_id="cardiffnlp/twitter-xlm-roberta-base",
    max_length=128
):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    results = {}

    for label in label_list:
        y_train_label = y_train[label].astype('category')
        y_val_label   = y_val[label].astype('category')
        y_test_label  = y_test[label].astype('category')

        y_train_codes = y_train_label.cat.codes.reset_index(drop=True)
        y_val_codes   = y_val_label.cat.codes.reset_index(drop=True)
        y_test_codes  = y_test_label.cat.codes.reset_index(drop=True)

        label_map = dict(enumerate(y_train_label.cat.categories))

        def create_dataset(X, y):
            df = pd.DataFrame({"text": X}).reset_index(drop=True)
            df["label"] = y
            dataset = Dataset.from_pandas(df)
            def tokenize(batch):
                enc = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length)
                enc["labels"] = batch["label"]
                return enc
            dataset = dataset.map(tokenize, batched=True)
            dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
            return dataset

        results[label] = {
            "train": create_dataset(X_train, y_train_codes),
            "val": create_dataset(X_val, y_val_codes),
            "test": create_dataset(X_test, y_test_codes),
            "num_labels": len(y_train_label.cat.categories),
            "label_map": label_map
        }

    return tokenizer, results

# ----------------------------
# Entrenamiento para una etiqueta
# ----------------------------
def train_model_for_label(train_dataset, val_dataset, num_labels,
                          model_id="cardiffnlp/twitter-xlm-roberta-base",
                          output_dir="./model_label",
                          epochs=3, batch_size=8, lr=2e-5):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=num_labels,
        problem_type="single_label_classification"
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        num_train_epochs=epochs,
        logging_steps=20,
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        report_to=[]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    best_f1 = 0
    best_checkpoint = None

    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        trainer.train()
        metrics = trainer.evaluate(val_dataset)
        print(f"Validation metrics: {metrics}")

        # Guardar manualmente el mejor modelo según F1
        if metrics["eval_f1"] > best_f1:
            best_f1 = metrics["eval_f1"]
            best_checkpoint = os.path.join(output_dir, f"best_model_epoch_{epoch+1}")
            trainer.save_model(best_checkpoint)
            print(f"Best model updated at epoch {epoch+1} with F1={best_f1:.4f}")

    print(f"\nTraining finished. Best F1: {best_f1:.4f}")
    return trainer, {"best_f1": best_f1, "best_checkpoint": best_checkpoint}


# ----------------------------
# Bloque principal
# ----------------------------
if __name__ == "__main__":
    import TextVectorRepresentation as TV

    path = "Datasets/EvaluationData/politicES_phase_2_train_public.csv"
    data = TV.load_data(path)
    data = data.sample(2000)  # Ajusta según tu memoria / tamaño de dataset

    train_data, val_data, test_data = TV.divide_train_val_test(data)
    X_train, y_train = TV.separate_x_y_vectors(train_data)
    X_val, y_val = TV.separate_x_y_vectors(val_data)
    X_test, y_test = TV.separate_x_y_vectors(test_data)

    labels = ["gender", "profession", "ideology_binary", "ideology_multiclass"]

    tokenizer, datasets = prepare_multilabel_datasets(
        X_train, y_train, X_val, y_val, X_test, y_test, labels
    )

    # Crear carpeta de modelos si no existe
    os.makedirs("./models", exist_ok=True)

    trainers = {}
    for label in labels:
        print(f"\nEntrenando modelo para {label}...")
        trainer, outputs = train_model_for_label(
            datasets[label]["train"],
            datasets[label]["val"],
            datasets[label]["num_labels"],
            output_dir=f"./models/{label}"
        )
        trainers[label] = trainer
        print(f"{label} - Best F1:", outputs["best_f1"])
        print(f"{label} - Checkpoint guardado en:", outputs["best_checkpoint"])

        # Evaluación en test
        test_results = trainer.evaluate(datasets[label]["test"])
        print(f"{label} - Test metrics:", test_results)
