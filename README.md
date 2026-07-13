# AI Learning Assistant — Emotion-Aware Support Platform

An end-to-end system that detects a student's emotional state from free-text descriptions of their study challenges, and responds with empathetic, personalized guidance. Built with BiLSTM and BERT emotion classifiers, Gemini-generated guidance, and full student/educator/admin authentication with a privacy-first data model.

**Live demo:** https://emotionlearning.streamlit.app

## What it does

- Takes a student's free-text description of a study problem
- Classifies the emotional state using two independently trained models:
  - **BiLSTM** (Keras/TensorFlow, trained from scratch)
  - **BERT** (fine-tuned DistilBERT via HuggingFace Transformers)
- Boosts predictions with rule-based keyword matching for extra reliability
- Detects mixed emotions (e.g. Curious + Confused) when scores are close
- Generates empathetic, actionable guidance using Google's Gemini API
- Logs every interaction to a SQLite database for analytics
- Displays trends and history in a built-in analytics dashboard
- Supports side-by-side model comparison, with an agreement/disagreement indicator

## Emotions detected

Bored, Confident, Confused, Curious, Frustrated

## Accounts & privacy

- **Students** sign up and verify their email via a real 6-digit code sent through Gmail SMTP
- **Educators** sign up the same way, but their account stays pending until an **admin** approves them
- A single admin account is auto-seeded on first run
- Each student picks an educator to link to at signup (changeable later from their Profile page), or opts to keep their interactions **private**
- Educators only ever see records from students linked to them — never anyone else's
- Admins can see that a private record exists (for oversight) but its content is masked
- Students can export their full data as JSON, or permanently delete their account, at any time

## Project structure

```
emotion-learning-assistant/
├── app.py                       # Streamlit entrypoint
├── requirements.txt
├── .env                         # local secrets (not committed)
├── reset_database.py            # wipes and re-seeds the database
├── cleanup_test_data.py         # removes leftover test accounts
├── fix_encoding.py               # fixes emoji mojibake if it recurs
├── test_educator_privacy.py      # automated: educator-linking & privacy
├── test_export_delete.py         # automated: data export & account deletion
├── test_email_verification.py    # automated: signup/verify/login flow
├── test_emoji_rendering.py       # automated: headless UI rendering check
├── test_real_models.py           # automated: BiLSTM/BERT prediction sanity check
├── data/
│   ├── dataset.csv               # generated training data
│   ├── train.csv / val.csv / test.csv
│   └── app.db                    # SQLite database (not committed)
├── models/
│   ├── bilstm/                   # trained BiLSTM model + tokenizer
│   └── bert/                     # fine-tuned BERT model + tokenizer (via Git LFS)
└── src/
    ├── config.py                 # loads secrets from .env or Streamlit Cloud secrets
    ├── database.py                # SQLite layer: users, auth, records, privacy
    ├── email_client.py            # Gmail SMTP verification emails
    ├── keyword_rules.py           # keyword-based emotion boosting
    ├── predict.py                  # unified prediction interface (loads both models)
    ├── gemini_client.py            # Gemini API wrapper
    ├── generate_dataset.py         # synthetic dataset generator (template-based)
    ├── prepare_goemotions.py       # dataset preparation helper
    ├── split_dataset.py            # train/val/test splitter
    ├── train_bilstm.py             # BiLSTM training script
    └── train_bert.py               # BERT fine-tuning script
```

## Setup

### 1. Clone and enter the project folder

```bash
git clone https://github.com/Hannu331/AI-emotion-learning.git
cd AI-emotion-learning
```

### 2. Create and activate a virtual environment (Python 3.10 or 3.11 required — TensorFlow has no wheels for 3.12+ or 3.14 yet)

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

```
GEMINI_API_KEY=your_real_gemini_key_here
GMAIL_ADDRESS=your_real_gmail@gmail.com
GMAIL_APP_PASSWORD=your_16_char_app_password
```

Get a Gemini key from [Google AI Studio](https://aistudio.google.com/apikey), and a Gmail App Password from `myaccount.google.com/apppasswords` (requires 2-Step Verification enabled on that Google account).

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
```
Email: admin@learningassistant.local
Password: ChangeMe123!
```

## Automated tests

Instead of manually clicking through the UI, these scripts verify core functionality directly:

```bash
python test_email_verification.py    # signup, code verification, login gating
python test_educator_privacy.py      # educator linking & privacy isolation
python test_export_delete.py         # data export & account deletion
python test_real_models.py           # BiLSTM/BERT prediction sanity checks
python test_emoji_rendering.py       # headless UI check via Streamlit AppTest
```

Utility scripts:

```bash
python reset_database.py       # full wipe, admin re-seeded fresh
python cleanup_test_data.py    # removes only leftover test_*@example.com accounts
python fix_encoding.py          # repairs emoji mojibake if it recurs after an edit
```

## Deployment (Streamlit Community Cloud)

The live version is deployed on Streamlit Community Cloud. Key setup notes if redeploying:

- **Python version must be pinned to 3.11** in the app's Advanced Settings — Streamlit Cloud's default (3.14 at time of writing) has no compatible TensorFlow wheels
- **Secrets** are set via the dashboard's Secrets box (TOML format), not `.env`:
  ```toml
  GEMINI_API_KEY = "..."
  GMAIL_ADDRESS = "..."
  GMAIL_APP_PASSWORD = "..."
  ```
- The BERT model (`model.safetensors`, ~268MB) is tracked via **Git LFS** — Streamlit Community Cloud supports LFS automatically, no extra config needed
- `requirements.txt` dependencies are version-pinned deliberately; unpinned installs can pull incompatible bleeding-edge releases and cause crashes

## How the dataset was built

No public dataset matched this project's exact 5-label academic-emotion taxonomy in text form (most public datasets in this space are facial/video-based, e.g. DAiSEE). A local template + slot-filling generator (`src/generate_dataset.py`) was built instead, mixing formal and casual phrasing per emotion, and validated against unseen freeform sentences to confirm the models generalize rather than just memorizing templates.

## Model comparison

BiLSTM and BERT are trained independently on the same data. The "Compare Both" mode in the app shows both models' predictions side by side and flags when they disagree — which tends to happen on genuinely ambiguous text.

## Known limitations

- Dataset is synthetically generated (template-based), not sourced from real student interactions
- Gemini free-tier quota limits real-time guidance generation to a small number of requests per day
- Emotion labels are single-label per interaction, though the keyword layer surfaces likely mixed emotions
- SQLite is used for simplicity; a production deployment at scale would want a hosted database instead
- Streamlit Community Cloud's free-tier memory (~1GB) is a real constraint when loading both TensorFlow and PyTorch models together
