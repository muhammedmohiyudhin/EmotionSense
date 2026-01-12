import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import Dataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

# Load cleaned dataset
df = pd.read_csv("../data/cleaned_emotions.csv")

labels = df["emotion"].unique().tolist()
label_map = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label_map.items()}

df["label"] = df["emotion"].map(label_map)

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained("saved_model")
tokenizer = BertTokenizer.from_pretrained("saved_model")

model.eval()

# Tokenize
def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding=True)

dataset = Dataset.from_pandas(df)
dataset = dataset.map(tokenize, batched=True)

# Use small sample for fast evaluation (optional)
dataset = dataset.shuffle(seed=42).select(range(2000))

true_labels = []
pred_labels = []

for item in dataset:
    inputs = {
        "input_ids": torch.tensor([item["input_ids"]]),
        "attention_mask": torch.tensor([item["attention_mask"]])
    }

    with torch.no_grad():
        outputs = model(**inputs)

    pred = torch.argmax(outputs.logits, dim=1).item()

    true_labels.append(item["label"])
    pred_labels.append(pred)

# Metrics
accuracy = accuracy_score(true_labels, pred_labels)
report = classification_report(true_labels, pred_labels, target_names=labels)
cm = confusion_matrix(true_labels, pred_labels)

# Save metrics
with open("../results/metrics.txt", "w") as f:
    f.write(f"Accuracy: {accuracy}\n\n")
    f.write("Classification Report:\n")
    f.write(report)

print("Accuracy:", accuracy)
print("\nClassification Report:\n", report)
print("\nConfusion Matrix:\n", cm)

print("\nMetrics saved in results/metrics.txt")
