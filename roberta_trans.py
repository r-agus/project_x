from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from init import xtrain, ytrain
from pandas import pandas as pd
from datasets import Dataset

# TRANSFORMERS LA PELÍCULA 
model_id = "cardiffnlp/twitter-xlm-roberta-base"

tokenizer = AutoTokenizer.from_pretrained(model_id)

df = pd.DataFrame({
    "tweet": xtrain,
    "labels": ytrain
}).dropna()

# Si labels NO son números, conviértelos:
if df["labels"].dtype == "object":
    df["labels"] = df["labels"].astype("category").cat.codes
    
dataset = Dataset.from_pandas(df)

def tokenize(batch): # TOKENIZA DESDE SU VOCABULARIO
    """
    Tokenize a batch of tweets with the project RoBERTa tokenizer.

    Args:
        batch (dict): Batch containing a ``"tweet"`` field.

    Returns:
        dict: Tokenized inputs ready for the Trainer API.
    """
    return tokenizer(batch["tweet"], truncation=True, padding="max_length", max_length=128)

dataset = dataset.map(tokenize, batched=True)

dataset.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])

train_test = dataset.train_test_split(test_size=0.2)

model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=df["labels"].nunique()
)

training_args = TrainingArguments(
    output_dir="./llama_gender",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=2e-5,
    num_train_epochs=2,
    evaluation_strategy="epoch",
    logging_steps=20,
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_test["train"],
    eval_dataset=train_test["test"]
)

outputs = trainer.train()
print(outputs)
