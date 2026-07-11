"""
Unified prediction interface. app.py only ever calls predict_emotion()
from this file — it doesn't care whether the underlying model is
BiLSTM or BERT under the hood.

Both models are now real, trained models (no placeholders):
- BiLSTM: models/bilstm/ (Keras)
- BERT: models/bert/ (fine-tuned distilbert-base-uncased)
"""

import os
import json
import pickle
from typing import Dict

import numpy as np
import torch
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.keyword_rules import EMOTIONS, apply_keyword_boost

BILSTM_DIR = "models/bilstm"
BERT_DIR = "models/bert"


class BiLSTMModel:
    """Loads the real trained BiLSTM model + tokenizer + config."""

    def __init__(self, model_dir: str):
        self.model = load_model(os.path.join(model_dir, "model.keras"))
        with open(os.path.join(model_dir, "tokenizer.pkl"), "rb") as f:
            self.tokenizer = pickle.load(f)
        with open(os.path.join(model_dir, "config.json")) as f:
            config = json.load(f)
        self.classes = config["classes"]
        self.max_len = config["max_len"]

    def predict(self, text: str) -> Dict[str, float]:
        seq = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=self.max_len, padding="post", truncating="post")
        probs = self.model.predict(padded, verbose=0)[0]
        return {cls: float(p) for cls, p in zip(self.classes, probs)}


class BERTModel:
    """Loads the fine-tuned BERT (distilbert) model + tokenizer + config."""

    def __init__(self, model_dir: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
        self.model.eval()
        with open(os.path.join(model_dir, "config_extra.json")) as f:
            config = json.load(f)
        self.classes = config["classes"]
        self.max_len = config["max_len"]

    def predict(self, text: str) -> Dict[str, float]:
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

        return {cls: float(p) for cls, p in zip(self.classes, probs)}


_bilstm_model = BiLSTMModel(BILSTM_DIR)
_bert_model = BERTModel(BERT_DIR)


def predict_emotion(text: str, model_choice: str = "bilstm") -> Dict[str, float]:
    """
    model_choice: "bilstm" or "bert"
    Returns keyword-boosted emotion probabilities, summing to 1.
    """
    if not text or not text.strip():
        return {emotion: 0.0 for emotion in EMOTIONS}

    model = _bilstm_model if model_choice == "bilstm" else _bert_model
    raw_scores = model.predict(text)
    return apply_keyword_boost(text, raw_scores)


def predict_both(text: str):
    """Used by the 'compare models' view in the app."""
    return {
        "bilstm": predict_emotion(text, "bilstm"),
        "bert": predict_emotion(text, "bert"),
    }