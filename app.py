"""
Streamlit entrypoint for the AI-Driven Emotion Detection & Personalized
Learning Support Platform. Handles: text input, emotion prediction
(single model or side-by-side comparison), Gemini-generated guidance,
CSV logging of every interaction, and a basic analytics dashboard.
"""

import os
import csv
from datetime import datetime

import streamlit as st
import pandas as pd

from src.predict import predict_emotion, predict_both
from src.gemini_client import generate_support_response_safe
from src.keyword_rules import EMOTIONS

LOG_PATH = "data/logs.csv"
LOG_FIELDS = ["timestamp", "student_text", "model_choice", "primary_emotion"] + EMOTIONS + ["gemini_response"]

st.set_page_config(page_title="AI Learning Assistant", layout="wide")


def ensure_log_file():
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            writer.writeheader()


def log_interaction(text: str, model_choice: str, scores: dict, gemini_response: str):
    ensure_log_file()
    primary = max(scores, key=scores.get)
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "student_text": text,
        "model_choice": model_choice,
        "primary_emotion": primary,
        "gemini_response": gemini_response,
    }
    row.update(scores)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        writer.writerow(row)

def render_scores(scores: dict, label: str):
    st.markdown(f"**{label}**")
    primary = max(scores, key=scores.get)
    confidence = scores[primary]
    st.markdown(f"Primary emotion: **{primary}** ({confidence:.0%})")

    if confidence >= 0.7:
        st.caption("High confidence")
    elif confidence >= 0.45:
        st.caption("Moderate confidence")
    else:
        st.caption("Low confidence — possibly mixed emotional state")

    df = pd.DataFrame(
        sorted(scores.items(), key=lambda kv: kv[1], reverse=True),
        columns=["Emotion", "Score"],
    )
    st.bar_chart(df.set_index("Emotion"))

    top_two = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:2]
    if len(top_two) == 2 and top_two[1][1] >= 0.25:
        st.caption(f"Mixed signal: also showing signs of **{top_two[1][0]}** ({top_two[1][1]:.0%})")


def get_support_tab():
    st.header("Describe Your Study Challenge")
    student_text = st.text_area(
        "What are you stuck on, or how are you feeling about it?",
        height=120,
        placeholder="e.g. I'm lost on recursion, nothing makes sense...",
    )

    col1, col2 = st.columns(2)
    with col1:
        mode = st.radio("Model", ["BiLSTM", "BERT", "Compare Both"], horizontal=True)
    with col2:
        show_ai = st.toggle("Show AI-generated guidance", value=True)

    btn_col1, btn_col2 = st.columns([1, 1])
    with btn_col1:
        analyze_clicked = st.button("Analyze", type="primary")
    with btn_col2:
        if st.button("Clear"):
            st.rerun()

    if analyze_clicked:
        if not student_text.strip():
            st.warning("Please enter some text first.")
            return

        if mode == "Compare Both":
            results = predict_both(student_text)
            c1, c2 = st.columns(2)
            with c1:
                render_scores(results["bilstm"], "BiLSTM")
            with c2:
                render_scores(results["bert"], "BERT")

            bilstm_primary = max(results["bilstm"], key=results["bilstm"].get)
            bert_primary = max(results["bert"], key=results["bert"].get)

            st.markdown("---")
            if bilstm_primary == bert_primary:
                st.success(f"Both models agree: **{bilstm_primary}**")
            else:
                st.warning(
                    f"Models disagree — BiLSTM says **{bilstm_primary}**, "
                    f"BERT says **{bert_primary}**. This text may reflect a mixed "
                    f"or ambiguous emotional state."
                )

            scores = results["bilstm"]
            model_choice = "bilstm+bert"
        else:
            model_choice = "bilstm" if mode == "BiLSTM" else "bert"
            scores = predict_emotion(student_text, model_choice)
            render_scores(scores, mode)

        gemini_response = ""
        if show_ai:
            st.markdown("---")
            st.subheader("Personalized Guidance")
            primary = max(scores, key=scores.get)
            mixed = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:3]
            with st.spinner("Generating guidance..."):
                gemini_response = generate_support_response_safe(student_text, primary, mixed)
            st.write(gemini_response)

        log_interaction(student_text, model_choice, scores, gemini_response)
        st.success("Interaction logged.")

def analytics_tab():
    st.header("Analytics Dashboard")
    if not os.path.exists(LOG_PATH):
        st.info("No interactions logged yet. Try the 'Get Support' tab first.")
        return

    df = pd.read_csv(LOG_PATH)
    if df.empty:
        st.info("No interactions logged yet.")
        return

    st.subheader("Emotion Frequency")
    counts = df["primary_emotion"].value_counts()
    st.bar_chart(counts)

    st.subheader("Emotion Trend Over Time")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    trend = df.groupby(["date", "primary_emotion"]).size().unstack(fill_value=0)
    st.line_chart(trend)

    st.subheader("Recent Interactions")
    st.dataframe(df.tail(10)[["timestamp", "student_text", "model_choice", "primary_emotion"]])


def main():
    st.title("AI Learning Assistant")
    st.caption("Emotion-aware support for learners and educators")
    tab1, tab2 = st.tabs(["Get Support", "Analytics Dashboard"])
    with tab1:
        get_support_tab()
    with tab2:
        analytics_tab()


if __name__ == "__main__":
    main()