# AI Learning Assistant — Emotion-Aware Support Platform

An end-to-end system that detects a student's emotional state from
free-text descriptions of their study challenges, and responds with
empathetic, personalized guidance.

## What it does

- Takes a student's free-text description of a study problem
- Classifies the emotional state using two independently trained models:
  - **BiLSTM** (Keras/TensorFlow, trained from scratch)
  - **BERT** (fine-tuned DistilBERT via HuggingFace Transformers)
- Boosts predictions with rule-based keyword matching for extra reliability
- Generates empathetic, actionable guidance using Google's Gemini API
- Logs every interaction to CSV for analytics
- Displays trends and history in a built-in analytics dashboard
- Supports side-by-side model comparison, with an agreement/disagreement indicator

## Emotions detected

Bored, Confident, Confused, Curious, Frustrated

## Project structure

emotion-learning-assistant/
├── app.py                     # Streamlit entrypoint
├── requirements.txt
├── .env                       # your Gemini API key (not committed)
├── data/
│   ├── dataset.csv            # generated training data
│   ├── train.csv / val.csv / test.csv
│   └── logs.csv               # interaction logs (auto-created)
├── models/
│   ├── bilstm/                # trained BiLSTM model + tokenizer
│   └── bert/                  # fine-tuned BERT model + tokenizer
└── src/
├── keyword_rules.py        # keyword-based emotion boosting
├── predict.py              # unified prediction interface (loads both models)
├── gemini_client.py        # Gemini API wrapper
├── generate_dataset.py     # synthetic dataset generator (template-based)
├── split_dataset.py        # train/val/test splitter
├── train_bilstm.py         # BiLSTM training script
└── train_bert.py           # BERT fine-tuning script

## Setup

### 1. Clone and enter the project folder

```bash
cd emotion-learning-assistant
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your Gemini API key

Create a `.env` file in the project root with:

GEMINI_API_KEY=your_real_key_here

Get a key from [Google AI Studio](https://aistudio.google.com/apikey).

> Note: the free tier has a daily quota (20 requests/day at time of
> writing). If you exceed it, the app gracefully falls back to a
> generic supportive message instead of crashing.

### 5. (Already done, but for reference) Generate data & train models

These steps were used to build the included `data/` and `models/`
folders. You don't need to re-run them unless retraining from scratch:

```bash
python -m src.generate_dataset
python -m src.split_dataset
python -m src.train_bilstm
python -m src.train_bert
```

### 6. Run the app

```bash
streamlit run app.py
```

Opens automatically at `http://localhost:8501`.

## How the dataset was built

No public dataset matched this project's exact 5-label academic-emotion
taxonomy in text form (most public datasets in this space are
facial/video-based, e.g. DAiSEE). The original plan was to generate
labeled examples using the Gemini API, but this hit free-tier daily
quota limits. Instead, a local template + slot-filling generator
(`src/generate_dataset.py`) was built, mixing formal and casual/
internet-style phrasing per emotion. This was validated by testing
trained models against completely unseen freeform sentences, confirming
the models generalize beyond memorized templates rather than just
recognizing the templates themselves.

## Model comparison

BiLSTM and BERT are trained independently on the same data. The
"Compare Both" mode in the app shows both models' predictions side by
side, and flags when they disagree — which tends to happen on
genuinely ambiguous text (e.g. text that could reasonably be read as
either Frustrated or Confused).

## Known limitations

- Dataset is synthetically generated (template-based), not sourced
  from real student interactions
- Gemini free-tier quota limits real-time guidance generation to a
  small number of requests per day
- Emotion labels are single-label per interaction; true student
  emotional states are often mixed