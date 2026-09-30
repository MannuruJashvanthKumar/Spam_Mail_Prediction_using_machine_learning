from pathlib import Path

import json
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "spam_mail_model.joblib"
VECTORIZER_PATH = BASE_DIR / "tfidf_vectorizer.joblib"
METRICS_PATH = BASE_DIR / "metrics.json"

st.set_page_config(
    page_title="MailGuard AI | Spam Mail Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------
# Theme / CSS
# ----------------------------
st.markdown(
    """
<style>
:root {
    --bg: #f5f7fb;
    --card: #ffffff;
    --ink: #172033;
    --muted: #687386;
    --line: #e7ebf2;
    --primary: #635bff;
    --primary-dark: #4b44d6;
    --danger: #dc3545;
    --danger-soft: #fff0f1;
    --success: #169b62;
    --success-soft: #ebfaf3;
    --shadow: 0 14px 36px rgba(31, 41, 55, 0.08);
}

html, body, [class*="css"] { font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
.stApp { background: var(--bg); }
.block-container { max-width: 1280px; padding-top: 1.5rem; padding-bottom: 3rem; }

.hero {
    background: linear-gradient(135deg, #171b3a 0%, #2e2b76 48%, #635bff 100%);
    border-radius: 26px;
    padding: 34px 36px;
    color: white;
    box-shadow: 0 18px 48px rgba(58, 48, 159, 0.24);
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hero:after {
    content: "";
    position: absolute;
    width: 260px;
    height: 260px;
    right: -80px;
    top: -120px;
    border-radius: 50%;
    background: rgba(255,255,255,0.09);
}
.hero-kicker { font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.14em; opacity: 0.78; font-weight: 700; }
.hero-title { font-size: 2.4rem; line-height: 1.05; font-weight: 800; margin: 8px 0; }
.hero-subtitle { max-width: 760px; font-size: 1.02rem; line-height: 1.6; opacity: 0.9; }

.card {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 20px;
    padding: 22px;
    box-shadow: var(--shadow);
    margin-bottom: 18px;
}
.card-title { font-size: 1rem; font-weight: 800; color: var(--ink); margin-bottom: 4px; }
.card-text { color: var(--muted); line-height: 1.55; }

.result-spam {
    background: var(--danger-soft);
    border: 1px solid #f6c9ce;
    color: #9f2330;
    border-radius: 18px;
    padding: 20px 22px;
}
.result-ham {
    background: var(--success-soft);
    border: 1px solid #bcebd4;
    color: #116d48;
    border-radius: 18px;
    padding: 20px 22px;
}
.result-label { font-size: 1.35rem; font-weight: 850; }
.result-note { margin-top: 5px; opacity: 0.84; }

.metric-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
.metric {
    background: #fff;
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 15px 16px;
}
.metric-value { font-size: 1.35rem; font-weight: 850; color: var(--ink); }
.metric-label { color: var(--muted); font-size: 0.82rem; margin-top: 2px; }

.badge {
    display: inline-flex; align-items: center; gap: 7px;
    border-radius: 999px; padding: 6px 11px;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.16);
    font-size: 0.78rem; font-weight: 700;
}
.small-muted { font-size: 0.84rem; color: var(--muted); }
.section-heading { color: var(--ink); font-size: 1.2rem; font-weight: 850; margin: 8px 0 12px; }
.footer { color: #8290a6; text-align: center; padding: 18px 0 4px; font-size: 0.82rem; }

[data-testid="stSidebar"] { background: #11152c; }
[data-testid="stSidebar"] * { color: #edf1ff !important; }
[data-testid="stSidebar"] .stRadio label { color: #edf1ff !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.12); }

div.stButton > button {
    border-radius: 12px;
    font-weight: 750;
}
div.stButton > button[kind="primary"] {
    background: var(--primary);
    border-color: var(--primary);
}
textarea { border-radius: 14px !important; }

@media (max-width: 800px) {
  .hero-title { font-size: 1.9rem; }
  .metric-row { grid-template-columns: 1fr; }
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource

def load_artifacts():
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            "Model artifacts are missing. Run `python train_model.py` first."
        )
    return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)


@st.cache_data

def load_metrics():
    if METRICS_PATH.exists():
        return pd.read_json(METRICS_PATH, typ="series")
    return pd.Series(dtype=object)


try:
    model, vectorizer = load_artifacts()
except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.stop()

metrics = load_metrics()

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.markdown("## 🛡️ MailGuard AI")
st.sidebar.caption("Spam mail detection powered by TF-IDF + Logistic Regression")
page = st.sidebar.radio("Workspace", ["Predict", "Analytics", "About"], index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("### Model status")
st.sidebar.markdown("🟢 **Trained model ready**")
st.sidebar.markdown("🟢 **TF-IDF vectorizer ready**")
st.sidebar.markdown(f"📦 **{int(metrics.get('dataset_rows', 5572)):,}** messages")
st.sidebar.markdown("---")
st.sidebar.caption("Educational / portfolio project. Predictions should be reviewed before making important decisions.")

# ----------------------------
# Hero
# ----------------------------
st.markdown(
    """
<div class="hero">
  <div class="hero-kicker">Machine Learning • NLP • Email Security</div>
  <div class="hero-title">MailGuard AI</div>
  <div class="hero-subtitle">Analyze an email or SMS-style message and estimate whether it is <b>spam</b> or <b>ham</b> using the trained model from your project dataset.</div>
  <div style="margin-top:16px; display:flex; gap:8px; flex-wrap:wrap;">
    <span class="badge">TF-IDF</span>
    <span class="badge">Logistic Regression</span>
    <span class="badge">Local inference</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)


if page == "Predict":
    st.markdown('<div class="section-heading">Prediction workspace</div>', unsafe_allow_html=True)

    col_input, col_info = st.columns([1.55, 0.85], gap="large")

    with col_input:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Paste your message</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-text">The model processes the message text and returns the predicted class with probability estimates.</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if "message" not in st.session_state:
            st.session_state.message = ""

        message = st.text_area(
            "Email / SMS message",
            key="message",
            height=260,
            placeholder="Example: Congratulations! You have won a free prize. Call now...",
            label_visibility="collapsed",
        )

        ex1, ex2, ex3, ex4 = st.columns(4)
        examples = {
            "Safe example": "Hey, are we still meeting at 7 tonight? Please send me the address.",
            "Spam example": "Congratulations! You have won a free cash prize. Call now to claim your reward!",
            "Promo example": "Limited offer! Get 70% off today only. Click the link to redeem your voucher.",
            "Work example": "Please review the attached project notes before tomorrow's meeting.",
        }
        buttons = [(ex1, "Safe example"), (ex2, "Spam example"), (ex3, "Promo example"), (ex4, "Work example")]
        for col, label in buttons:
            with col:
                if st.button(label, use_container_width=True):
                    st.session_state.message = examples[label]
                    st.rerun()

        analyze = st.button("🔍 Analyze message", type="primary", use_container_width=True)

        if analyze:
            if not message.strip():
                st.warning("Enter a message before running the prediction.")
            else:
                features = vectorizer.transform([message])
                prediction = int(model.predict(features)[0])
                probabilities = model.predict_proba(features)[0]
                # Model classes are encoded as spam=0 and ham=1 in training.
                class_to_prob = {int(c): float(p) for c, p in zip(model.classes_, probabilities)}
                spam_probability = class_to_prob.get(0, 0.0)
                ham_probability = class_to_prob.get(1, 0.0)

                if prediction == 0:
                    st.markdown(
                        f"""
<div class="result-spam">
  <div class="result-label">🚨 Likely SPAM</div>
  <div class="result-note">The trained classifier assigns a higher probability to the spam class.</div>
</div>
""",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
<div class="result-ham">
  <div class="result-label">✅ Likely HAM (Not Spam)</div>
  <div class="result-note">The trained classifier assigns a higher probability to the ham class.</div>
</div>
""",
                        unsafe_allow_html=True,
                    )

                st.write("")
                p1, p2 = st.columns(2)
                with p1:
                    st.markdown(
                        f"""
<div class="metric"><div class="metric-value">{spam_probability:.1%}</div><div class="metric-label">Spam probability</div></div>
""",
                        unsafe_allow_html=True,
                    )
                    st.progress(spam_probability)
                with p2:
                    st.markdown(
                        f"""
<div class="metric"><div class="metric-value">{ham_probability:.1%}</div><div class="metric-label">Ham probability</div></div>
""",
                        unsafe_allow_html=True,
                    )
                    st.progress(ham_probability)

                confidence = max(spam_probability, ham_probability)
                st.markdown(
                    f"<div class='small-muted'>Model confidence (highest class probability): <b>{confidence:.1%}</b></div>",
                    unsafe_allow_html=True,
                )

                with st.expander("Why did the model lean this way?", expanded=False):
                    st.caption("This is an approximate feature-level explanation for the Logistic Regression model, not a causal explanation.")
                    row = features.toarray()[0]
                    names = vectorizer.get_feature_names_out()
                    coefficients = model.coef_[0]
                    contributions = row * coefficients
                    nonzero = contributions != 0
                    detail = pd.DataFrame({
                        "term": names[nonzero],
                        "contribution": contributions[nonzero],
                    })
                    if detail.empty:
                        st.info("No vocabulary terms from the trained TF-IDF vocabulary were found in this message.")
                    else:
                        spam_terms = detail.sort_values("contribution").head(8).copy()
                        ham_terms = detail.sort_values("contribution", ascending=False).head(8).copy()
                        left, right = st.columns(2)
                        with left:
                            st.markdown("**Terms leaning toward spam**")
                            st.dataframe(spam_terms.assign(contribution=spam_terms["contribution"].round(3)), hide_index=True, use_container_width=True)
                        with right:
                            st.markdown("**Terms leaning toward ham**")
                            st.dataframe(ham_terms.assign(contribution=ham_terms["contribution"].round(3)), hide_index=True, use_container_width=True)

    with col_info:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">How it works</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-text">Your text is converted into TF-IDF features. A Logistic Regression classifier then estimates the two class probabilities.</div>', unsafe_allow_html=True)
        st.markdown("<br><b>Pipeline</b><br>Message → TF-IDF → Logistic Regression → Spam / Ham", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        dataset_rows = int(metrics.get("dataset_rows", 0))
        test_acc = float(metrics.get("test_accuracy", 0))
        st.markdown(
            f"""
<div class="metric-row" style="grid-template-columns:1fr 1fr; margin-bottom:18px;">
  <div class="metric"><div class="metric-value">{dataset_rows:,}</div><div class="metric-label">Dataset messages</div></div>
  <div class="metric"><div class="metric-value">{test_acc:.2%}</div><div class="metric-label">Test accuracy</div></div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Tips for better checks</div>', unsafe_allow_html=True)
        st.markdown(
            """
<div class="card-text">
• Paste the complete message when possible.<br>
• Preserve links, unusual punctuation, and promotional wording.<br>
• Treat model probability as an estimate, not proof.<br>
• For important messages, verify the sender independently.
</div>
""",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)


elif page == "Analytics":
    st.markdown('<div class="section-heading">Model & dataset analytics</div>', unsafe_allow_html=True)
    metric_cols = st.columns(4)
    analytics = [
        ("Test accuracy", float(metrics.get("test_accuracy", 0)), "{:.2%}"),
        ("Spam precision", float(metrics.get("precision_spam", 0)), "{:.2%}"),
        ("Spam recall", float(metrics.get("recall_spam", 0)), "{:.2%}"),
        ("Spam F1", float(metrics.get("f1_spam", 0)), "{:.2%}"),
    ]
    for col, (label, value, fmt) in zip(metric_cols, analytics):
        with col:
            st.markdown(
                f"<div class='metric'><div class='metric-value'>{fmt.format(value)}</div><div class='metric-label'>{label}</div></div>",
                unsafe_allow_html=True,
            )

    st.write("")
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Dataset composition</div>', unsafe_allow_html=True)
        counts = pd.DataFrame({
            "Category": ["Ham", "Spam"],
            "Messages": [int(metrics.get("ham_count", 4825)), int(metrics.get("spam_count", 747))],
        }).set_index("Category")
        st.bar_chart(counts)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Confusion matrix</div>', unsafe_allow_html=True)
        cm = metrics.get("confusion_matrix", [[118, 31], [1, 965]])
        cm_df = pd.DataFrame(cm, index=["Actual Spam", "Actual Ham"], columns=["Predicted Spam", "Predicted Ham"])
        st.dataframe(cm_df, use_container_width=True)
        st.caption("The matrix follows the training label encoding: spam = 0 and ham = 1.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Model details</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
<div class="card-text">
<b>Algorithm:</b> Logistic Regression<br>
<b>Text representation:</b> TF-IDF vectorization with English stop-word removal<br>
<b>Split:</b> 80% training / 20% test, stratified, random_state=3<br>
<b>Training accuracy:</b> {float(metrics.get('training_accuracy', 0)):.2%}
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


else:
    st.markdown('<div class="section-heading">About this project</div>', unsafe_allow_html=True)
    about1, about2 = st.columns([1.2, 0.8], gap="large")
    with about1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Project objective</div>', unsafe_allow_html=True)
        st.markdown(
            "<div class='card-text'>Build a practical machine-learning system that learns from labeled messages in the project dataset and predicts whether a new message belongs to the spam or ham class.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<br><div class='card-title'>Source workflow</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='card-text'>The project preserves the supplied notebook's main workflow: CSV loading, missing-value handling, label encoding, train/test split, TF-IDF text features, Logistic Regression training, and evaluation.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with about2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Project resources</div>', unsafe_allow_html=True)
        st.markdown(
            """
<div class="card-text">
📄 README and full documentation<br>
🧪 Reproducible training script<br>
📓 Clean training notebook<br>
💾 Trained model + TF-IDF vectorizer<br>
📊 Metrics and evaluation resources<br>
🗂️ Original supplied notebook
</div>
""",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.info("This application is an educational/project demonstration. A classification result is an automated estimate and should not be treated as definitive proof that a message is safe or malicious.")

st.markdown("<div class='footer'>MailGuard AI • Built from the supplied spam-mail dataset and training workflow</div>", unsafe_allow_html=True)
