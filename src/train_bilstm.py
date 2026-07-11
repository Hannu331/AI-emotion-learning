"""
Trains a BiLSTM emotion classifier on data/train.csv, validates on
data/val.csv, and saves the trained model + tokenizer vocabulary to
models/bilstm/ for later use in predict.py.

Run from project root:
    python -m src.train_bilstm
"""

import os
import json
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from sklearn.preprocessing import LabelEncoder

TRAIN_PATH = "data/train.csv"
VAL_PATH = "data/val.csv"
MODEL_DIR = "models/bilstm"

MAX_VOCAB = 5000
MAX_LEN = 40
EMBEDDING_DIM = 64
LSTM_UNITS = 64
EPOCHS = 30
BATCH_SIZE = 16


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["emotion"])
    y_val = label_encoder.transform(val_df["emotion"])

    tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_df["text"])

    X_train = pad_sequences(
        tokenizer.texts_to_sequences(train_df["text"]),
        maxlen=MAX_LEN, padding="post", truncating="post"
    )
    X_val = pad_sequences(
        tokenizer.texts_to_sequences(val_df["text"]),
        maxlen=MAX_LEN, padding="post", truncating="post"
    )

    num_classes = len(label_encoder.classes_)
    vocab_size = min(MAX_VOCAB, len(tokenizer.word_index) + 1)

    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=MAX_LEN),
        Bidirectional(LSTM(LSTM_UNITS)),
        Dropout(0.4),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=5, restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1,
    )

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nFinal validation accuracy: {val_acc:.4f}")

    # Save model
    model.save(os.path.join(MODEL_DIR, "model.keras"))

    # Save tokenizer
    with open(os.path.join(MODEL_DIR, "tokenizer.pkl"), "wb") as f:
        pickle.dump(tokenizer, f)

    # Save label encoder classes + config
    with open(os.path.join(MODEL_DIR, "config.json"), "w") as f:
        json.dump({
            "classes": label_encoder.classes_.tolist(),
            "max_len": MAX_LEN,
        }, f, indent=2)

    print(f"\nSaved model, tokenizer, and config to {MODEL_DIR}/")


if __name__ == "__main__":
    main()