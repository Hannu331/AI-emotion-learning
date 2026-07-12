import src.config  # noqa: F401  (loads secrets for both local .env and cloud deployment)
import os
import csv
import json
from datetime import datetime

import streamlit as st
import pandas as pd
from src.database import create_user, authenticate_user, verify_email, resend_verification_code,get_records_for_user, get_all_records, get_pending_educators,set_user_status, get_active_educators, set_linked_educator,get_records_for_educator, get_all_records_for_admin,export_user_data, delete_user_account
from src.predict import predict_emotion, predict_both
from src.gemini_client import generate_support_response_safe
from src.keyword_rules import EMOTIONS
from src.database import (
    init_db, create_user, authenticate_user, insert_emotion_record,
    get_records_for_user, get_all_records, get_pending_educators, set_user_status,
)

LOG_PATH = "data/logs.csv"
LOG_FIELDS = ["timestamp", "student_text", "model_choice", "primary_emotion"] + EMOTIONS + ["gemini_response"]

st.set_page_config(page_title="AI Learning Assistant", page_icon="🧠", layout="wide")
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

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

    st.markdown('<div class="emotion-card">', unsafe_allow_html=True)
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


def login_screen():
    st.markdown(
        """
        <div class="hero">
            <h1>🧠 AI Learning Assistant</h1>
            <p>Emotion-aware support for learners and educators</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_login, tab_signup = st.tabs(["Log In", "Sign Up"])

    with tab_login:
        st.subheader("Log in to your account")
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Log In", type="primary"):
            if not email or not password:
                st.warning("Please enter both email and password.")
            else:
                success, user, message = authenticate_user(email, password)
                if success:
                    st.session_state["user"] = user
                    st.rerun()
                else:
                    st.error(message)

    with tab_signup:
        st.subheader("Create a new account")

        if "pending_verification_email" not in st.session_state:
            st.session_state.pending_verification_email = None

        if st.session_state.pending_verification_email:
            # Show the verification-code entry screen
            pending_email = st.session_state.pending_verification_email
            st.info(f"A verification code was sent to **{pending_email}**. Enter it below.")
            code_input = st.text_input("6-digit verification code", key="verify_code_input", max_chars=6)

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Verify", type="primary"):
                    ok, msg = verify_email(pending_email, code_input)
                    if ok:
                        st.success(msg)
                        st.session_state.pending_verification_email = None
                    else:
                        st.error(msg)
            with col2:
                if st.button("Resend code"):
                    ok, msg = resend_verification_code(pending_email)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)

            if st.button("Cancel / use a different email"):
                st.session_state.pending_verification_email = None
                st.rerun()

        else:
            # Show the normal signup form
            name = st.text_input("Name", key="signup_name")
            signup_email = st.text_input("Email", key="signup_email")
            role = st.selectbox("Role", ["student", "educator"], key="signup_role")
            if role == "educator":
                st.caption("⚠️ Educator accounts require administrator approval before you can log in.")

            selected_educator_email = None
            if role == "student":
                educators = get_active_educators()
                educator_options = ["No educator (keep private)"] + [
                    f"{e['name']} ({e['email']})" for e in educators
                ]
                educator_choice = st.selectbox("Select your educator", educator_options, key="signup_educator")
                if educator_choice != "No educator (keep private)":
                    selected_educator_email = educator_choice.split("(")[-1].rstrip(")")
                if not educators:
                    st.caption("No approved educators yet — you can stay private and link one later from your profile.")

            signup_password = st.text_input("Password", type="password", key="signup_password")

            if st.button("Sign Up", type="primary"):
                if not name or not signup_email or not signup_password:
                    st.warning("Please fill in all fields.")
                else:
                    success, message = create_user(signup_email, name, signup_password, role)
                    if success:
                        if role == "student":
                            set_linked_educator(signup_email, selected_educator_email)
                        st.success(message)
                        st.session_state.pending_verification_email = signup_email
                        st.rerun()
                    else:
                        st.error(message)


def admin_panel():
    st.subheader("🛡️ Administrator Panel")
    st.caption("Review and approve pending educator account requests")

    pending = get_pending_educators()
    if not pending:
        st.info("No pending educator requests.")
    else:
        for user in pending:
            with st.container(border=True):
                c1, c2, c3 = st.columns([3, 1, 1])
                with c1:
                    st.write(f"**{user['name']}** — {user['email']}")
                    st.caption(f"Requested: {user['created_at']}")
                with c2:
                    if st.button("Approve", key=f"approve_{user['email']}"):
                        set_user_status(user["email"], "active")
                        st.rerun()
                with c3:
                    if st.button("Reject", key=f"reject_{user['email']}"):
                        set_user_status(user["email"], "rejected")
                        st.rerun()
    st.divider()
    st.markdown("#### All interaction records (private entries masked)")
    records = get_all_records_for_admin()
    if not records:
        st.info("No interactions logged yet.")
    else:
        import pandas as pd
        df = pd.DataFrame(records)
        display_cols = ["timestamp", "email", "field", "input_text", "predicted_emotion", "model_used"]
        st.dataframe(df[display_cols], use_container_width=True)


def get_support_tab():
    st.subheader("💬 Describe your study challenge")

    field = st.selectbox(
        "Academic field",
        ["Computer Science", "Mathematics", "Science", "Language/Writing",
         "History/Social Studies", "Other"],
    )

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

        primary = max(scores, key=scores.get)
        top_two = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:2]
        secondary = top_two[1][0] if len(top_two) == 2 else None

        insert_emotion_record(
            email=st.session_state["user"]["email"],
            field=field,
            input_text=student_text,
            predicted_emotion=primary,
            secondary_emotion=secondary,
            confidence_score=scores[primary],
            model_used=model_choice,
            ai_response=gemini_response,
            response_type="gemini" if gemini_response and "unavailable" not in gemini_response else "fallback_template",
            emotion_scores_json=json.dumps(scores),
            csv_logged=True,
        )
        st.toast("Interaction logged ✅")


def analytics_tab():
    st.subheader("📊 Analytics dashboard")

    user = st.session_state["user"]

    if user["role"] == "educator":
        records = get_records_for_educator(user["email"])
        st.caption("Showing data for students linked to you")
    else:
        records = get_records_for_user(user["email"])
        st.caption("Showing your own interaction history")

    if not records:
        st.info("No interactions logged yet. Try the 'Get Support' tab first.")
        return

    df = pd.DataFrame(records)

    m1, m2, m3 = st.columns(3)
    m1.metric("Total interactions", len(df))
    m2.metric("Most common emotion", df["predicted_emotion"].value_counts().idxmax())
    m3.metric("Unique emotions seen", df["predicted_emotion"].nunique())

    st.markdown("#### Emotion frequency")
    counts = df["predicted_emotion"].value_counts()
    st.bar_chart(counts, color="#A78BFA")

    st.markdown("#### Emotion trend over time")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    trend = df.groupby(["date", "predicted_emotion"]).size().unstack(fill_value=0)
    st.line_chart(trend)

    st.markdown("#### Recent interactions")
    display_cols = ["timestamp", "email", "field", "input_text", "model_used", "predicted_emotion"] \
        if user["role"] == "educator" else ["timestamp", "field", "input_text", "model_used", "predicted_emotion"]
    st.dataframe(df.head(10)[display_cols], use_container_width=True)
def profile_tab():
    st.subheader("👤 Your Profile")
    user = st.session_state["user"]
    st.write(f"**Name:** {user['name']}")
    st.write(f"**Email:** {user['email']}")
    st.write(f"**Role:** {user['role']}")

    if user["role"] == "student":
        st.markdown("#### Educator link")
        st.caption("Choose who can see your interaction history, or keep it private.")

        educators = get_active_educators()
        educator_options = ["No educator (keep private)"] + [
            f"{e['name']} ({e['email']})" for e in educators
        ]

        current = user.get("linked_educator")
        current_label = "No educator (keep private)"
        for e in educators:
            if e["email"] == current:
                current_label = f"{e['name']} ({e['email']})"
                break

        default_index = educator_options.index(current_label) if current_label in educator_options else 0

def main():
    init_db()

    if "user" not in st.session_state:
        login_screen()
        return

    user = st.session_state["user"]

    st.markdown(
        f"""
        <div class="hero">
            <h1>🧠 AI Learning Assistant</h1>
            <p>Welcome back, {user['name']} · logged in as {user['role']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Log Out"):
        del st.session_state["user"]
        st.rerun()

    if user["role"] == "admin":
        admin_panel()
        return

    tab1, tab2, tab3 = st.tabs(["💬 Get Support", "📊 Analytics Dashboard", "👤 Profile"])
    with tab1:
        get_support_tab()
    with tab2:
        analytics_tab()
    with tab3:
        profile_tab()


if __name__ == "__main__":
    main()
