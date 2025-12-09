from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import pandas as pd
import torch

def prepare_gender_dataset(X_train, y_train, X_val, y_val, X_test, y_test,
                           model_id="cardiffnlp/twitter-xlm-roberta-base", max_length=128):
    """
    Prepara datasets para la columna 'Gender' con multiclass (male, female, non-binary).
    Convierte labels a enteros.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Convertir a categoría y luego a códigos enteros
    y_train = y_train['gender'].astype('category').cat.codes.reset_index(drop=True)
    y_val = y_val['gender'].astype('category').cat.codes.reset_index(drop=True)
    y_test = y_test['gender'].astype('category').cat.codes.reset_index(drop=True)

    label_map = dict(enumerate(y_train.astype('category').cat.categories)) if hasattr(y_train, 'cat') else None

    def create_dataset(X, y):
        df = pd.DataFrame({"text": X}).reset_index(drop=True)
        df['label'] = y
        dataset = Dataset.from_pandas(df)

        def tokenize(batch):
            encodings = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length)
            encodings["labels"] = batch["label"]
            return encodings

        dataset = dataset.map(tokenize, batched=True)
        dataset.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])
        return dataset

    train_dataset = create_dataset(X_train, y_train)
    val_dataset = create_dataset(X_val, y_val)
    test_dataset = create_dataset(X_test, y_test)

    num_labels = len(y_train.unique())

    return tokenizer, train_dataset, val_dataset, test_dataset, num_labels


def train_gender_model(train_dataset, val_dataset, num_labels,
                       model_id="cardiffnlp/twitter-xlm-roberta-base",
                       output_dir="./roberta_gender", epochs=3, batch_size=8, lr=2e-5):
    """
    Entrena RoBERTa para predecir la columna Gender.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=num_labels,
        problem_type="single_label_classification"  # multiclass
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
        eval_dataset=val_dataset
    )

    outputs = trainer.train()
    return trainer, outputs


if __name__ == "__main__":
    import TextVectorRepresentation as TV

    path = "Datasets/EvaluationData/politicES_phase_2_train_public.csv"
    data = TV.load_data(path)
    data = data.head(500)  # Reducimos filas para pruebas

    # Dividir en train, val, test
    train_data, val_data, test_data = TV.divide_train_val_test(data)

    # Separar X e y
    X_train, y_train = TV.separate_x_y_vectors(train_data)
    X_val, y_val = TV.separate_x_y_vectors(val_data)
    X_test, y_test = TV.separate_x_y_vectors(test_data)

    # Preparar datasets
    tokenizer, train_dataset, val_dataset, test_dataset, num_labels = prepare_gender_dataset(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    # Entrenar
    trainer, outputs = train_gender_model(train_dataset, val_dataset, num_labels)

    # Resultados
    print("Training loss:", outputs.training_loss)
    print("Global steps:", outputs.global_step)
