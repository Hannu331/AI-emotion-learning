# AI Learning Assistant — Emotion-Aware Support Platform

An end-to-end system that detects a student's emotional state from free-text descriptions of their study challenges, and responds with empathetic, personalized guidance. Built with BiLSTM and BERT emotion classifiers, Gemini-generated guidance, and full student/educator/admin authentication.

## What it does

- Takes a student's free-text description of a study problem
- Classifies the emotional state using two independently trained models:
  - **BiLSTM** (Keras/TensorFlow, trained from scratch)
  - **BERT** (fine-tuned DistilBERT via HuggingFace Transformers)
- Boosts predictions with rule-based keyword matching for extra reliability
- Generates empathetic, actionable guidance using Google's Gemini API
- Logs every interaction to a SQLite database for analytics
- Displays trends and history in a built-in analytics dashboard
- Supports side-by-side model comparison, with an agreement/disagreement indicator
- Full authentication: student/educator signup with real Gmail email verification, admin approval workflow for educators, and student-to-educator linking with a private/self option

## Emotions detected

Bored, Confident, Confused, Curious, Frustrated

## Project structure
emotion-learning-assistant/
├── app.py                     # Streamlit entrypoint
├── requirements.txt
├── .env                       # your Gemini/Gmail secrets (not committed)
├── data/
│   ├── dataset.csv            # generated training data
│   ├── train.csv / val.csv / test.csv
│   └── app.db                 # SQLite database (not committed)
├── models/
│   ├── bilstm/                # trained BiLSTM model + tokenizer
│   └── bert/                  # fine-tuned BERT model + tokenizer (via Git LFS)
└── src/
├── config.py               # loads secrets from .env or Streamlit Cloud secrets
├── database.py             # SQLite layer: users, auth, records, privacy
├── email_client.py         # Gmail SMTP verification emails
├── keyword_rules.py        # keyword-based emotion boosting
├── predict.py               # unified prediction interface (loads both models)
├── gemini_client.py         # Gemini API wrapper
├── generate_dataset.py      # synthetic dataset generator (template-based)
├── split_dataset.py         # train/val/test splitter
├── train_bilstm.py          # BiLSTM training script
└── train_bert.py            # BERT fine-tuning script

## Setup

### 1. Clone and enter the project folder

```bash
git clone https://github.com/Hannu331/AI-emotion-learning.git
cd AI-emotion-learning
```

### 2. Create and activate a virtual environment (Python 3.10 or 3.11 required)

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your secrets

Create a `.env` file in the project root:

GEMINI_API_KEY=AQ.Ab8RN6J9DKoY9BafjHFRWdZmZZ-iG7wm1egqpGGU22guD4A6FQ
GMAIL_ADDRESS=emotionassistantlearningai@gmail.com
GMAIL_APP_PASSWORD=tfyo rmju pwrc vwqh

Get a Gemini key from [Google AI Studio](https://aistudio.google.com/apikey), and a Gmail App Password from `myaccount.google.com/apppasswords` (requires 2-Step Verification enabled).

### 5. (Already done, but for reference) Generate data & train models

These steps built the included `data/` and `models/` folders. You don't need to re-run them unless retraining from scratch:

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

Default admin login (seeded automatically on first run):

Email: admin@learningassistant.local
Password: ChangeMe123!

## How the dataset was built

No public dataset matched this project's exact 5-label academic-emotion taxonomy in text form (most public datasets in this space are facial/video-based, e.g. DAiSEE). A local template + slot-filling generator (`src/generate_dataset.py`) was built instead, mixing formal and casual phrasing per emotion, and validated against unseen freeform sentences to confirm the models generalize rather than just memorizing templates.

## Model comparison

BiLSTM and BERT are trained independently on the same data. The "Compare Both" mode in the app shows both models' predictions side by side and flags when they disagree — which tends to happen on genuinely ambiguous text.

## Privacy model

Students choose an educator to link to at signup (changeable later from their Profile), or opt to keep their interactions fully private. Educators only see records from students linked to them. Admins can see that a private record exists (for oversight) but its content is masked. Students can export their full data as JSON or permanently delete their account at any time.

## Known limitations

- Dataset is synthetically generated (template-based), not sourced from real student interactions
- Gemini free-tier quota limits real-time guidance generation to a small number of requests per day
- Emotion labels are single-label per interaction, though the keyword layer surfaces likely mixed emotions
- SQLite is used for simplicity; a production deployment would want a hosted database instead