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

st.set_page_config(page_title="AI Emotion & Learning Assistant", page_icon="🧠", layout="wide")

CUSTOM_CSS = """
<style>
.stApp {
    background: linear-gradient(180deg, #FAFAFC 0%, #F5F3FF 100%);
}
.hero {
    padding: 2rem 2rem 1.5rem 2rem;
    border-radius: 16px;
    background: linear-gradient(135deg, #7C5CFC 0%, #A78BFA 100%);
    color: white;
    margin-bottom: 1.5rem;
}
.hero h1 {
    margin: 0;
    font-size: 2rem;
    font-weight: 700;
}
.hero p {
    margin-top: 0.4rem;
    opacity: 0.9;
    font-size: 1rem;
}
.emotion-card {
    padding: 1rem 1.2rem;
    border-radius: 12px;
    background: white;
    border: 1px solid #EAE6FB;
    box-shadow: 0 2px 8px rgba(124, 92, 252, 0.08);
    margin-bottom: 0.5rem;
}
.emotion-badge {
    display: inline-block;
    padding: 0.25rem 0.8rem;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.95rem;
    color: white;
}
.badge-Bored { background: #9CA3AF; }
.badge-Confident { background: #22C55E; }
.badge-Confused { background: #F59E0B; }
.badge-Curious { background: #3B82F6; }
.badge-Frustrated { background: #EF4444; }
div[data-testid="stMetric"] {
    background: white;
    border-radius: 12px;
    padding: 0.8rem;
    border: 1px solid #EAE6FB;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


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


def emotion_badge(emotion: str, confidence: float):
    st.markdown(
        f'<span class="emotion-badge badge-{emotion}">{emotion} · {confidence:.0%}</span>',
        unsafe_allow_html=True,
    )


def render_scores(scores: dict, label: str):
    primary = max(scores, key=scores.get)
    confidence = scores[primary]

    st.markdown(f'<div class="emotion-card">', unsafe_allow_html=True)
    st.markdown(f"**{label}**")
    emotion_badge(primary, confidence)

    if confidence >= 0.7:
        st.caption("🟢 High confidence")
    elif confidence >= 0.45:
        st.caption("🟡 Moderate confidence")
    else:
        st.caption("🟠 Low confidence — possibly mixed emotional state")

    df = pd.DataFrame(
        sorted(scores.items(), key=lambda kv: kv[1], reverse=True),
        columns=["Emotion", "Score"],
    )
    st.bar_chart(df.set_index("Emotion"), color="#7C5CFC")

    top_two = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:2]
    if len(top_two) == 2 and top_two[1][1] >= 0.25:
        st.caption(f"Also showing signs of **{top_two[1][0]}** ({top_two[1][1]:.0%})")
    st.markdown("</div>", unsafe_allow_html=True)


def get_support_tab():
    st.subheader("💬 Describe your study challenge")
    student_text = st.text_area(
        "What are you stuck on, or how are you feeling about it?",
        height=120,
        placeholder="e.g. I'm lost on recursion, nothing makes sense...",
        label_visibility="collapsed",
    )

    col1, col2 = st.columns(2)
    with col1:
        mode = st.radio("Model", ["BiLSTM", "BERT", "Compare Both"], horizontal=True)
    with col2:
        show_ai = st.toggle("Show AI-generated guidance", value=True)

    btn_col1, btn_col2 = st.columns([1, 1])
    with btn_col1:
        analyze_clicked = st.button("✨ Analyze", type="primary", use_container_width=True)
    with btn_col2:
        if st.button("Clear", use_container_width=True):
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

            if bilstm_primary == bert_primary:
                st.success(f"✅ Both models agree: **{bilstm_primary}**")
            else:
                st.warning(
                    f"⚖️ Models disagree — BiLSTM says **{bilstm_primary}**, "
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
            st.subheader("🌱 Personalized guidance")
            primary = max(scores, key=scores.get)
            mixed = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:3]
            with st.spinner("Generating guidance..."):
                gemini_response = generate_support_response_safe(student_text, primary, mixed)
            st.info(gemini_response)

        log_interaction(student_text, model_choice, scores, gemini_response)
        st.toast("Interaction logged ✅")


def analytics_tab():
    st.subheader("📊 Analytics dashboard")
    if not os.path.exists(LOG_PATH):
        st.info("No interactions logged yet. Try the 'Get Support' tab first.")
        return

    df = pd.read_csv(LOG_PATH)
    if df.empty:
        st.info("No interactions logged yet.")
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("Total interactions", len(df))
    m2.metric("Most common emotion", df["primary_emotion"].value_counts().idxmax())
    m3.metric("Unique emotions seen", df["primary_emotion"].nunique())

    st.markdown("#### Emotion frequency")
    counts = df["primary_emotion"].value_counts()
    st.bar_chart(counts, color="#A78BFA")

    st.markdown("#### Emotion trend over time")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    trend = df.groupby(["date", "primary_emotion"]).size().unstack(fill_value=0)
    st.line_chart(trend)

    st.markdown("#### Recent interactions")
    st.dataframe(df.tail(10)[["timestamp", "student_text", "model_choice", "primary_emotion"]], use_container_width=True)


def main():
    st.markdown(
        """
        <div class="hero">
            <h1>🧠 AI Emotion & Learning Assistant</h1>
            <p>Emotion-aware support for learners and educators</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    tab1, tab2 = st.tabs(["💬 Get Support", "📊 Analytics Dashboard"])
    with tab1:
        get_support_tab()
    with tab2:
        analytics_tab()


if __name__ == "__main__":
    main()