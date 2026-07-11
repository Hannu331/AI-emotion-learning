"""
Fine-tunes a pretrained BERT model (distilbert-base-uncased, for speed
on CPU) on data/train.csv, validates on data/val.csv, and saves the
fine-tuned model + tokenizer to models/bert/ for use in predict.py.

Run from project root:
    python -m src.train_bert
"""

import os
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.preprocessing import LabelEncoder

TRAIN_PATH = "data/train.csv"
VAL_PATH = "data/val.csv"
MODEL_DIR = "models/bert"

BASE_MODEL = "distilbert-base-uncased"  # smaller/faster than full BERT, good for CPU
MAX_LEN = 40
BATCH_SIZE = 16
EPOCHS = 4
LR = 2e-5


class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["emotion"])
    y_val = label_encoder.transform(val_df["emotion"])
    num_classes = len(label_encoder.classes_)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=num_classes
    ).to(device)

    train_ds = EmotionDataset(train_df["text"].tolist(), y_train, tokenizer, MAX_LEN)
    val_ds = EmotionDataset(val_df["text"].tolist(), y_val, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} - train_loss: {avg_train_loss:.4f} - val_accuracy: {val_acc:.4f}")

    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    with open(os.path.join(MODEL_DIR, "config_extra.json"), "w") as f:
        json.dump({"classes": label_encoder.classes_.tolist(), "max_len": MAX_LEN}, f, indent=2)

    print(f"\nSaved fine-tuned BERT model to {MODEL_DIR}/")


if __name__ == "__main__":
    main()