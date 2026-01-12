import pandas as pd
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset

# -----------------------------
# Load cleaned dataset
# -----------------------------
df = pd.read_csv("../data/cleaned_emotions.csv")

# Encode labels
labels = df["emotion"].unique().tolist()
label_map = {label: i for i, label in enumerate(labels)}
df["label"] = df["emotion"].map(label_map)

print("Emotion labels:", label_map)

# -----------------------------
# Tokenizer
# -----------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True)

# -----------------------------
# Dataset conversion
# -----------------------------
dataset = Dataset.from_pandas(df)
dataset = dataset.map(tokenize, batched=True)

dataset = dataset.train_test_split(test_size=0.2)

# -----------------------------
# Data collator for padding
# -----------------------------
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# -----------------------------
# Model
# -----------------------------
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(labels)
)

# -----------------------------
# Training configuration
# -----------------------------
training_args = TrainingArguments(
    output_dir="../results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    logging_dir="../results/logs"
)

# -----------------------------
# Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator
)

# -----------------------------
# Train
# -----------------------------
trainer.train()

# -----------------------------
# Save model
# -----------------------------
model.save_pretrained("saved_model")
tokenizer.save_pretrained("saved_model")

print("Training completed successfully. Model saved in model/saved_model/")

