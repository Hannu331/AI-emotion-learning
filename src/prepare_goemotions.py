"""
Builds data/dataset.csv from real GoEmotions data (for Confused,
Curious, Frustrated, Confident) combined with synthetic template-based
examples for Bored (GoEmotions has no boredom label).

GoEmotions label -> our label mapping:
    confusion  -> Confused
    curiosity  -> Curious
    annoyance  -> Frustrated
    pride, admiration -> Confident

Only single-label rows are used (rows with exactly one emotion tag),
to avoid ambiguous multi-label examples. Classes are capped to the
smallest available class size so the final dataset stays balanced.

Run from project root:
    python -m src.prepare_goemotions
"""

import os
import csv
import random

random.seed(42)

RAW_DIR = "data/goemotions_raw/data"
OUTPUT_PATH = "data/dataset.csv"

SOURCE_TO_TARGET = {
    "confusion": "Confused",
    "curiosity": "Curious",
    "annoyance": "Frustrated",
    "pride": "Confident",
    "admiration": "Confident",
}

TARGET_LABELS = ["Bored", "Confident", "Confused", "Curious", "Frustrated"]

# --- Bored templates (reused from generate_dataset.py, since GoEmotions
# has no boredom label) ---
SUBJECTS = [
    "recursion", "this calculus problem", "the essay outline", "photosynthesis",
    "this Python error", "the group project plan", "linear algebra", "this lab report",
    "the history reading", "SQL joins", "this grammar rule", "thermodynamics",
    "the coding assignment", "statistics homework", "this chemistry equation",
    "the research paper thesis", "object-oriented programming", "this proof",
    "the presentation slides", "machine learning basics", "this French verb tense",
    "the biology diagram", "algorithms and complexity", "this economics concept",
    "the exam material", "data structures", "this physics problem set",
    "this loop", "the assignment instructions", "binary search", "this API",
    "the lecture notes", "my code", "this formula", "the reading assignment",
]
CONTEXT_TAGS = [
    "for my assignment", "before the exam", "in today's lecture", "for the group project",
    "while revising my draft", "during lab", "for homework", "in class today",
    "before the deadline", "while studying alone", "", "", "right now", "again",
]
BORED_TEMPLATES = [
    "I've been staring at {subject} {context} and I just can't focus anymore.",
    "Honestly {subject} feels so repetitive, I keep zoning out.",
    "Not gonna lie, {subject} is putting me to sleep {context}.",
    "I don't feel motivated to keep going through {subject} right now.",
    "This is dragging on forever, {subject} just isn't interesting to me.",
    "I keep checking the clock instead of working on {subject}.",
    "{subject} feels like it's going nowhere, I've lost interest.",
    "Meh, {subject} again... I've done this a hundred times.",
    "I can't bring myself to care about {subject} today.",
    "Everything about {subject} feels flat and uninteresting right now.",
    "I'm just going through the motions with {subject}, not really engaged.",
    "Same old stuff with {subject}, nothing new is grabbing my attention.",
    "ugh {subject} again, so boring",
    "meh another {subject}, we already know this stuff",
    "cant focus on {subject} at all rn, so dull",
    "yawn... {subject} is not holding my attention today",
    "smh {subject} is dragging on forever",
    "not even a little interested in {subject} tbh",
    "zoned out like 3 times reading about {subject}",
    "{subject} is just... nothing new, kinda bored",
    "why is {subject} so dry, i can barely stay awake",
    "skimming through {subject} bc its so uninteresting",
    "this {subject} lecture is a snoozefest",
    "i keep drifting off thinking about literally anything else during {subject}",
]


def load_emotion_names():
    path = os.path.join(RAW_DIR, "emotions.txt")
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_split(filename, emotion_names, rows_out):
    path = os.path.join(RAW_DIR, filename)
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            text, label_str = parts[0], parts[1]
            label_ids = label_str.split(",")
            if len(label_ids) != 1:
                continue  # skip multi-label rows for clean single-emotion mapping
            try:
                idx = int(label_ids[0])
            except ValueError:
                continue
            if idx >= len(emotion_names):
                continue
            source_label = emotion_names[idx]
            target = SOURCE_TO_TARGET.get(source_label)
            if target:
                text = text.strip()
                if text:
                    rows_out.append({"text": text, "emotion": target})


def generate_bored(n):
    combos = set()
    results = []
    attempts = 0
    while len(results) < n and attempts < n * 30:
        attempts += 1
        template = random.choice(BORED_TEMPLATES)
        subject = random.choice(SUBJECTS)
        context = random.choice(CONTEXT_TAGS)
        text = template.format(subject=subject, context=context)
        text = " ".join(text.split()).replace(" .", ".")
        key = (template, subject, context)
        if key not in combos:
            combos.add(key)
            results.append(text)
    return results


def main():
    emotion_names = load_emotion_names()
    print(f"Loaded {len(emotion_names)} GoEmotions label names")

    rows = []
    for split_file in ["train.tsv", "dev.tsv", "test.tsv"]:
        before = len(rows)
        load_split(split_file, emotion_names, rows)
        print(f"{split_file}: +{len(rows) - before} mapped rows (running total {len(rows)})")

    # Count per target label (excluding Bored, which isn't in GoEmotions)
    from collections import Counter
    counts = Counter(r["emotion"] for r in rows if r["emotion"] != "Bored")
    print("\nRaw mapped counts (before balancing):")
    for label in ["Confident", "Confused", "Curious", "Frustrated"]:
        print(f"  {label}: {counts.get(label, 0)}")

    # Cap each of the 4 GoEmotions-derived classes to the smallest class size
    cap = min(counts.get(label, 0) for label in ["Confident", "Confused", "Curious", "Frustrated"])
    cap = min(cap, 300)  # also cap at 300 max per class to keep training fast
    print(f"\nBalancing all 4 GoEmotions-derived classes to {cap} examples each")

    by_label = {"Confident": [], "Confused": [], "Curious": [], "Frustrated": []}
    for r in rows:
        if r["emotion"] in by_label:
            by_label[r["emotion"]].append(r)

    balanced_rows = []
    for label, items in by_label.items():
        random.shuffle(items)
        balanced_rows.extend(items[:cap])

    # Generate synthetic Bored examples to match the same class size
    bored_texts = generate_bored(cap)
    for text in bored_texts:
        balanced_rows.append({"text": text, "emotion": "Bored"})

    random.shuffle(balanced_rows)

    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "emotion"])
        writer.writeheader()
        writer.writerows(balanced_rows)

    print(f"\nDone. Saved {len(balanced_rows)} examples to {OUTPUT_PATH}")
    final_counts = Counter(r["emotion"] for r in balanced_rows)
    print("Final class distribution:", dict(final_counts))


if __name__ == "__main__":
    main()
