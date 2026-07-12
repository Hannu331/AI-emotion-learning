"""
Sends real verification emails via Gmail SMTP, using an app password
(not your actual Gmail password) stored in .env.

Requires in .env:
    GMAIL_ADDRESS=your_real_gmail@gmail.com
    GMAIL_APP_PASSWORD=your16charapppassword
"""

import os
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv

load_dotenv()

GMAIL_ADDRESS = os.getenv("GMAIL_ADDRESS")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465


def send_verification_email(to_email: str, code: str) -> tuple[bool, str]:
    if not GMAIL_ADDRESS or not GMAIL_APP_PASSWORD:
        return False, "Email sending is not configured (missing GMAIL_ADDRESS / GMAIL_APP_PASSWORD in .env)."

    subject = "Verify your AI Learning Assistant account"
    body = (
        f"Hi,\n\n"
        f"Your verification code is: {code}\n\n"
        f"Enter this code in the app to verify your email address. "
        f"This code expires in 10 minutes.\n\n"
        f"If you didn't request this, you can ignore this email."
    )

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = GMAIL_ADDRESS
    msg["To"] = to_email

    try:
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as server:
            server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
            server.sendmail(GMAIL_ADDRESS, [to_email], msg.as_string())
        return True, "Verification email sent."
    except Exception as e:
        return False, f"Failed to send email: {e}"
