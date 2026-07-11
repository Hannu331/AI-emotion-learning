"""
Wrapper around the Gemini API (new google-genai SDK). Takes a student's
text + their detected emotion(s) and returns empathetic, actionable
guidance.
"""

import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-3.5-flash"

_client = None


def _get_client():
    global _client
    if _client is None:
        if not API_KEY or API_KEY == "your_gemini_api_key_here":
            raise ValueError(
                "GEMINI_API_KEY is not set. Add your real key to .env"
            )
        _client = genai.Client(api_key=API_KEY)
    return _client


def build_prompt(student_text: str, primary_emotion: str, mixed_emotions: list) -> str:
    emotion_summary = ", ".join(f"{e} ({s:.0%})" for e, s in mixed_emotions)
    return f"""You are a warm, encouraging academic support assistant.

A student wrote this about a challenge they're facing:
"{student_text}"

Detected emotional state: {primary_emotion} (primary)
Full emotion breakdown: {emotion_summary}

Write a short, empathetic response with:
1. A one-sentence acknowledgement of how they feel (no clinical language, sound human).
2. Two or three concrete, actionable next steps tailored to their described problem.
3. One short line of genuine encouragement.

Keep it under 120 words total. Do not repeat the emotion label itself in the response."""


def generate_support_response(student_text: str, primary_emotion: str, mixed_emotions: list) -> str:
    """
    Returns Gemini's generated guidance text. Raises on failure so the
    caller (app.py) can decide how to degrade gracefully.
    """
    client = _get_client()
    prompt = build_prompt(student_text, primary_emotion, mixed_emotions)
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )
    return response.text.strip()


def generate_support_response_safe(student_text: str, primary_emotion: str, mixed_emotions: list) -> str:
    """
    Same as above, but never raises — falls back to a generic message
    if the API key is missing or the call fails.
    """
    try:
        return generate_support_response(student_text, primary_emotion, mixed_emotions)
    except Exception as e:
        return (
            f"(AI response unavailable: {e})\n\n"
            f"Based on your {primary_emotion.lower()} state, try breaking the "
            f"problem into smaller pieces and revisiting the fundamentals "
            f"before moving forward. You've got this."
        )