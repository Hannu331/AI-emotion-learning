"""
Splits data/dataset.csv into train/val/test CSVs (stratified by emotion
so each split keeps the same class balance). Saves to:
    data/train.csv
    data/val.csv
    data/test.csv

Run from project root:
    python -m src.split_dataset
"""

import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_PATH = "data/dataset.csv"
TRAIN_PATH = "data/train.csv"
VAL_PATH = "data/val.csv"
TEST_PATH = "data/test.csv"

TRAIN_FRAC = 0.7
VAL_FRAC = 0.15
TEST_FRAC = 0.15


def main():
    df = pd.read_csv(INPUT_PATH)
    print("Loaded", len(df), "total examples")

    train_val, test = train_test_split(
        df, test_size=TEST_FRAC, stratify=df["emotion"], random_state=42
    )
    val_frac_of_remainder = VAL_FRAC / (TRAIN_FRAC + VAL_FRAC)
    train, val = train_test_split(
        train_val, test_size=val_frac_of_remainder, stratify=train_val["emotion"], random_state=42
    )

    train.to_csv(TRAIN_PATH, index=False)
    val.to_csv(VAL_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)

    print("Train:", len(train), "saved to", TRAIN_PATH)
    print("Val:", len(val), "saved to", VAL_PATH)
    print("Test:", len(test), "saved to", TEST_PATH)

    print()
    print("Train class distribution:")
    print(train["emotion"].value_counts())


if __name__ == "__main__":
    main()