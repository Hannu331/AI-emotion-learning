"""
SQLite database layer for the AI Learning Assistant. Implements the
Users and Emotion_Records schema from the ER diagram, plus an
account-approval workflow:

- Students can self-signup and log in immediately (status='active').
- Educators can sign up, but their account is created with
  status='pending' and cannot log in until an admin approves it.
- A single admin account is seeded automatically on first run.
"""

import os
import sqlite3
import hashlib
import uuid
from datetime import datetime
import random
from datetime import timedelta
from src.email_client import send_verification_email

DB_PATH = "data/app.db"

ADMIN_EMAIL = "admin@learningassistant.local"
ADMIN_DEFAULT_PASSWORD = "ChangeMe123!"  # change this after first login


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def init_db():
    os.makedirs("data", exist_ok=True)
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS Users (
            email TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'student',
            status TEXT NOT NULL DEFAULT 'active',
            login_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS Emotion_Records (
            record_id TEXT PRIMARY KEY,
            email TEXT NOT NULL,
            field TEXT,
            input_text TEXT NOT NULL,
            predicted_emotion TEXT NOT NULL,
            secondary_emotion TEXT,
            confidence_score REAL,
            model_used TEXT,
            ai_response TEXT,
            response_type TEXT,
            emotion_scores TEXT,
            timestamp TEXT NOT NULL,
            csv_logged INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY (email) REFERENCES Users(email)
        )
    """)

    # Migration safety: if an older Users table exists without `status`,
    # add the column so existing databases don't break.
    c.execute("PRAGMA table_info(Users)")
    existing_cols = {row["name"] for row in c.fetchall()}
    if "status" not in existing_cols:
        c.execute("ALTER TABLE Users ADD COLUMN status TEXT NOT NULL DEFAULT 'active'")
    if "email_verified" not in existing_cols:
        c.execute("ALTER TABLE Users ADD COLUMN email_verified INTEGER NOT NULL DEFAULT 0")
    if "verification_code" not in existing_cols:
        c.execute("ALTER TABLE Users ADD COLUMN verification_code TEXT")
    if "verification_code_expires" not in existing_cols:
        c.execute("ALTER TABLE Users ADD COLUMN verification_code_expires TEXT")
    if "linked_educator" not in existing_cols:
        c.execute("ALTER TABLE Users ADD COLUMN linked_educator TEXT")

    conn.commit()

    # Seed a single admin account if one doesn't already exist.
    c.execute("SELECT email FROM Users WHERE email = ?", (ADMIN_EMAIL,))
    if c.fetchone() is None:
        c.execute(
            "INSERT INTO Users (email, name, password, role, status, login_count, created_at, email_verified) "
            "VALUES (?, ?, ?, 'admin', 'active', 0, ?, 1)",
            (ADMIN_EMAIL, "Administrator", _hash_password(ADMIN_DEFAULT_PASSWORD), datetime.now().isoformat()),
        )
        conn.commit()
    else:
        # Admin existed before email_verified was added — force it verified.
        c.execute("UPDATE Users SET email_verified = 1 WHERE email = ?", (ADMIN_EMAIL,))
        conn.commit()

    conn.close()



def _generate_code():
    return f"{random.randint(0, 999999):06d}"


def create_user(email: str, name: str, password: str, role: str = "student"):
    """
    All new accounts require email verification before login, regardless
    of role. Educators additionally require admin approval, which only
    becomes relevant after they verify their email.
    """
    if role == "admin":
        return False, "Admin accounts cannot be created through signup."

    status = "active" if role == "student" else "pending"
    code = _generate_code()
    expires = (datetime.now() + timedelta(minutes=15)).isoformat()

    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO Users (email, name, password, role, status, login_count, "
            "created_at, email_verified, verification_code, verification_code_expires) "
            "VALUES (?, ?, ?, ?, ?, 0, ?, 0, ?, ?)",
            (email, name, _hash_password(password), role, status,
             datetime.now().isoformat(), code, expires),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return False, "An account with this email already exists."
    conn.close()

    sent, send_msg = send_verification_email(email, code)
    if not sent:
        return True, f"Account created, but the verification email failed to send: {send_msg}"
    return True, "Account created. Check your email for a 6-digit verification code."
def get_active_educators():
    """Returns approved educators for the signup/profile dropdown."""
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT email, name FROM Users WHERE role = 'educator' AND status = 'active' ORDER BY name")
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def set_linked_educator(student_email: str, educator_email):
    """educator_email can be a real email, or None for 'self' (private)."""
    conn = get_connection()
    c = conn.cursor()
    c.execute("UPDATE Users SET linked_educator = ? WHERE email = ?", (educator_email, student_email))
    conn.commit()
    conn.close()


def get_records_for_educator(educator_email: str):
    """Only records belonging to students who linked this educator."""
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        """
        SELECT er.* FROM Emotion_Records er
        JOIN Users u ON er.email = u.email
        WHERE u.linked_educator = ?
        ORDER BY er.timestamp DESC
        """,
        (educator_email,),
    )
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows

def get_all_records_for_admin():
    """
    Admin sees every record exists, but content is masked for students
    who have no linked_educator (private/self).
    """
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        """
        SELECT er.*, u.linked_educator FROM Emotion_Records er
        JOIN Users u ON er.email = u.email
        ORDER BY er.timestamp DESC
        """
    )
    rows = [dict(r) for r in c.fetchall()]
    conn.close()

    for r in rows:
        if not r["linked_educator"]:
            r["input_text"] = "🔒 Private"
            r["ai_response"] = "🔒 Private"
            r["predicted_emotion"] = "🔒 Private"
            r["secondary_emotion"] = None
            r["emotion_scores"] = None
    return rows

def verify_email(email: str, code: str):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM Users WHERE email = ?", (email,))
    row = c.fetchone()
    if row is None:
        conn.close()
        return False, "No account found with this email."
    if row["email_verified"]:
        conn.close()
        return True, "Email already verified. You can log in."
    if row["verification_code"] != code:
        conn.close()
        return False, "Incorrect verification code."
    if datetime.now().isoformat() > row["verification_code_expires"]:
        conn.close()
        return False, "This code has expired. Please request a new one."

    c.execute(
        "UPDATE Users SET email_verified = 1, verification_code = NULL, "
        "verification_code_expires = NULL WHERE email = ?", (email,)
    )
    conn.commit()
    conn.close()
    return True, "Email verified successfully. You can now log in."


def resend_verification_code(email: str):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM Users WHERE email = ?", (email,))
    row = c.fetchone()
    if row is None:
        conn.close()
        return False, "No account found with this email."
    if row["email_verified"]:
        conn.close()
        return False, "Email is already verified."

    code = _generate_code()
    expires = (datetime.now() + timedelta(minutes=15)).isoformat()
    c.execute(
        "UPDATE Users SET verification_code = ?, verification_code_expires = ? WHERE email = ?",
        (code, expires, email),
    )
    conn.commit()
    conn.close()

    sent, send_msg = send_verification_email(email, code)
    if not sent:
        return False, f"Failed to send email: {send_msg}"
    return True, "A new verification code was sent to your email."

def authenticate_user(email: str, password: str):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM Users WHERE email = ?", (email,))
    row = c.fetchone()
    if row is None:
        conn.close()
        return False, None, "No account found with this email."
    if row["password"] != _hash_password(password):
        conn.close()
        return False, None, "Incorrect password."

    if not row["email_verified"]:
        conn.close()
        return False, None, "Please verify your email before logging in."
    if row["status"] == "rejected":
        conn.close()
        return False, None, "Your account request was not approved. Contact the administrator for details."

    c.execute("UPDATE Users SET login_count = login_count + 1 WHERE email = ?", (email,))
    conn.commit()
    c.execute("SELECT * FROM Users WHERE email = ?", (email,))
    updated_row = c.fetchone()
    conn.close()
    return True, dict(updated_row), "Login successful."


def get_pending_educators():
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM Users WHERE role = 'educator' AND status = 'pending' ORDER BY created_at")
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def set_user_status(email: str, status: str):
    conn = get_connection()
    c = conn.cursor()
    c.execute("UPDATE Users SET status = ? WHERE email = ?", (status, email))
    conn.commit()
    conn.close()


def get_all_users():
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT email, name, role, status, login_count, created_at FROM Users ORDER BY created_at")
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows

def export_user_data(email: str):
    """
    Returns a dict with the user's profile (no password) and all their
    emotion records — for a GDPR-style data export.
    """
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM Users WHERE email = ?", (email,))
    user_row = c.fetchone()
    if user_row is None:
        conn.close()
        return None

    user_dict = dict(user_row)
    user_dict.pop("password", None)
    user_dict.pop("verification_code", None)
    user_dict.pop("verification_code_expires", None)

    c.execute("SELECT * FROM Emotion_Records WHERE email = ? ORDER BY timestamp DESC", (email,))
    records = [dict(r) for r in c.fetchall()]
    conn.close()

    return {"profile": user_dict, "emotion_records": records}

def delete_user_account(email: str):
    """
    Permanently deletes a user and all their emotion records.
    Cannot be used to delete the seeded admin account.
    """
    if email == ADMIN_EMAIL:
        return False, "The administrator account cannot be deleted."

    conn = get_connection()
    c = conn.cursor()
    c.execute("DELETE FROM Emotion_Records WHERE email = ?", (email,))
    c.execute("DELETE FROM Users WHERE email = ?", (email,))
    conn.commit()
    conn.close()
    return True, "Account and all associated data have been permanently deleted."


def insert_emotion_record(
    email, field, input_text, predicted_emotion, secondary_emotion,
    confidence_score, model_used, ai_response, response_type,
    emotion_scores_json, csv_logged=False,
):
    conn = get_connection()
    c = conn.cursor()
    record_id = str(uuid.uuid4())
    c.execute(
        """INSERT INTO Emotion_Records
        (record_id, email, field, input_text, predicted_emotion, secondary_emotion,
         confidence_score, model_used, ai_response, response_type, emotion_scores,
         timestamp, csv_logged)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            record_id, email, field, input_text, predicted_emotion, secondary_emotion,
            confidence_score, model_used, ai_response, response_type, emotion_scores_json,
            datetime.now().isoformat(), int(csv_logged),
        ),
    )
    conn.commit()
    conn.close()
    return record_id


def get_records_for_user(email: str):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM Emotion_Records WHERE email = ? ORDER BY timestamp DESC", (email,))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def get_all_records():
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM Emotion_Records ORDER BY timestamp DESC")
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows
