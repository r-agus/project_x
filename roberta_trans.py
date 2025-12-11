from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


# ============================================================
# PREPARAR DATASETS
# ============================================================
def prepare_gender_dataset(X_train, y_train, X_val, y_val, X_test, y_test,
                           model_id="cardiffnlp/twitter-xlm-roberta-base", max_length=128):

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Convertimos gender a números
    y_train = y_train['gender'].astype('category')
    y_val = y_val['gender'].astype('category')
    y_test = y_test['gender'].astype('category')

    label2id = {v: i for i, v in enumerate(y_train.cat.categories)}
    id2label = {i: v for v, i in label2id.items()}

    y_train = y_train.map(label2id).reset_index(drop=True)
    y_val = y_val.map(label2id).reset_index(drop=True)
    y_test = y_test.map(label2id).reset_index(drop=True)

    def create_dataset(X, y):
        df = pd.DataFrame({"text": X})
        df["label"] = y
        dataset = Dataset.from_pandas(df)

        def tokenize(batch):
            enc = tokenizer(
                batch["text"],
                truncation=True,
                padding="max_length",
                max_length=max_length
            )
            enc["labels"] = batch["label"]
            return enc

        dataset = dataset.map(tokenize, batched=True)
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        return dataset

    return (
        tokenizer,
        create_dataset(X_train, y_train),
        create_dataset(X_val, y_val),
        create_dataset(X_test, y_test),
        len(label2id),
        label2id,
        id2label
    )


# ============================================================
# ENTRENAR MODELO
# ============================================================
def train_gender_model(train_dataset, val_dataset, num_labels,
                       model_id="cardiffnlp/twitter-xlm-roberta-base",
                       output_dir="./roberta_gender", epochs=3, batch_size=8, lr=2e-5):

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
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=20,
        report_to=[],
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


# ============================================================
# EVALUACIÓN COMPLETA
# ============================================================
def evaluate_model(trainer, test_dataset, id2label):
    print("\n🔵 Evaluando en TEST SET...")

    preds_output = trainer.predict(test_dataset)
    
    preds = preds_output.predictions.argmax(axis=1)
    labels = preds_output.label_ids

    print("\n📌 Accuracy:", accuracy_score(labels, preds))
    print("📌 F1-macro:", f1_score(labels, preds, average="macro"))
    print("📌 F1-weighted:", f1_score(labels, preds, average="weighted"))

    print("\n📌 Classification Report:")
    print(classification_report(labels, preds, target_names=[id2label[i] for i in range(len(id2label))]))

    print("\n📌 Confusion Matrix:")
    print(confusion_matrix(labels, preds))


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    import TextVectorRepresentation as TV

    path = "Datasets/EvaluationData/politicES_phase_2_train_public.csv"
    data = TV.load_data(path)
    data = data.head(500)

    train_data, val_data, test_data = TV.divide_train_val_test(data)

    X_train, y_train = TV.separate_x_y_vectors(train_data)
    X_val, y_val = TV.separate_x_y_vectors(val_data)
    X_test, y_test = TV.separate_x_y_vectors(test_data)

    tokenizer, train_ds, val_ds, test_ds, num_labels, label2id, id2label = prepare_gender_dataset(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    trainer, outputs = train_gender_model(train_ds, val_ds, num_labels)

    print("\nTraining loss:", outputs.training_loss)
    print("Global steps:", outputs.global_step)

    # 🔥 EVALUACIÓN COMPLETA
    evaluate_model(trainer, test_ds, id2label)
