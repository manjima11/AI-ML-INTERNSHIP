"""
Streamlit App — CIFAR-10 CNN Image Classifier
Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os, pathlib

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_PATH = "cnn_model.h5"
CLASS_NAMES = [
    "✈️ Airplane", "🚗 Automobile", "🐦 Bird", "🐱 Cat", "🦌 Deer",
    "🐶 Dog",      "🐸 Frog",       "🐴 Horse", "🚢 Ship", "🚛 Truck"
]
CLASS_RAW = [c.split(" ", 1)[1] for c in CLASS_NAMES]   # strip emoji

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CNN Image Classifier",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.main-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6366f1, #8b5cf6, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.subtitle { color: #94a3b8; font-weight: 300; font-size: 1rem; margin-top: 0; }

.pred-card {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
    border: 1px solid #6366f1;
    border-radius: 16px;
    padding: 1.6rem 2rem;
    margin-top: 1.2rem;
    box-shadow: 0 8px 32px rgba(99,102,241,0.25);
}
.pred-label { font-family: 'Space Mono', monospace; color: #a5b4fc; font-size: 0.78rem; letter-spacing: 0.12em; text-transform: uppercase; }
.pred-class { font-family: 'Space Mono', monospace; color: #e0e7ff; font-size: 2rem; font-weight: 700; margin: 0.3rem 0; }
.pred-conf  { color: #6ee7b7; font-size: 1.1rem; font-weight: 600; }

.info-box {
    background: #0f172a;
    border-left: 3px solid #6366f1;
    border-radius: 8px;
    padding: 0.9rem 1.2rem;
    font-size: 0.88rem;
    color: #94a3b8;
    margin-top: 0.8rem;
}
.bar-wrap { margin-top: 1.2rem; }
.bar-row   { display: flex; align-items: center; gap: 10px; margin: 5px 0; font-size: 0.84rem; }
.bar-label { width: 120px; color: #cbd5e1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.bar-bg    { flex: 1; background: #1e293b; border-radius: 4px; height: 10px; }
.bar-fill  { height: 10px; border-radius: 4px; background: linear-gradient(90deg,#6366f1,#a78bfa); }
.bar-pct   { width: 46px; text-align: right; color: #94a3b8; font-family: 'Space Mono', monospace; font-size: 0.78rem; }

.stButton>button {
    background: linear-gradient(135deg,#6366f1,#8b5cf6);
    color: white; border: none; border-radius: 10px;
    padding: 0.55rem 1.6rem; font-weight: 600; font-size: 0.95rem;
    cursor: pointer; transition: opacity .2s;
}
.stButton>button:hover { opacity: 0.88; }
</style>
""", unsafe_allow_html=True)

# ── Load model (cached) ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    if not pathlib.Path(MODEL_PATH).exists():
        return None
    return tf.keras.models.load_model(MODEL_PATH)

# ── Preprocess uploaded image ──────────────────────────────────────────────────
def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((32, 32), Image.LANCZOS)
    arr = np.array(img, dtype="float32") / 255.0
    return np.expand_dims(arr, 0)          # (1, 32, 32, 3)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 About the Model")
    st.markdown("""
**Architecture:** Custom CNN  
**Dataset:** CIFAR-10  
**Input size:** 32 × 32 × 3  
**Classes:** 10  
**Framework:** TensorFlow / Keras

---
**Network Blocks**
- 3 × Conv blocks (32→64→128 filters)
- Batch Normalization after each Conv
- MaxPooling for spatial reduction
- Dropout for regularisation
- Dense(512) classifier head

---
**Training**
- Optimizer: Adam (lr=1e-3)
- Loss: Categorical Cross-entropy
- Augmentation: flip, rotate, zoom
- EarlyStopping + ReduceLROnPlateau
""")

    st.markdown("---")
    st.markdown("**Supported classes**")
    for c in CLASS_NAMES:
        st.markdown(f"- {c}")

# ── Main ───────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">CNN Image Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">CIFAR-10 · TensorFlow · Streamlit</p>', unsafe_allow_html=True)
st.markdown("---")

model = load_model()

if model is None:
    st.error(
        "⚠️ **Model file not found.**\n\n"
        "Run `python cnn_model.py` first to train and save `cnn_model.h5`, "
        "then relaunch this app."
    )
    st.info("**Quick start:**\n```bash\npython cnn_model.py\nstreamlit run app.py\n```")
    st.stop()

st.success("✅ Model loaded successfully!", icon="🚀")

# Upload
uploaded = st.file_uploader(
    "Upload an image (JPG / PNG / WEBP)",
    type=["jpg", "jpeg", "png", "webp"],
    help="The model was trained on 32×32 CIFAR-10 images — it works best with clear, centered subjects."
)

if uploaded:
    col1, col2 = st.columns([1, 1.6], gap="large")

    with col1:
        pil_img = Image.open(uploaded)
        st.image(pil_img, caption="Uploaded Image", use_container_width=True)

    with col2:
        with st.spinner("Running inference…"):
            inp   = preprocess(pil_img)
            probs = model.predict(inp, verbose=0)[0]          # (10,)
            top_i = int(np.argmax(probs))
            top_p = float(probs[top_i])

        # Prediction card
        st.markdown(f"""
<div class="pred-card">
  <div class="pred-label">Predicted Class</div>
  <div class="pred-class">{CLASS_NAMES[top_i]}</div>
  <div class="pred-conf">Confidence: {top_p*100:.1f}%</div>
</div>
""", unsafe_allow_html=True)

        # Bar chart for all classes
        st.markdown('<div class="bar-wrap"><b style="color:#e2e8f0">All class probabilities</b>', unsafe_allow_html=True)
        sorted_idx = np.argsort(probs)[::-1]
        bars_html = ""
        for i in sorted_idx:
            pct   = probs[i] * 100
            width = f"{pct:.1f}%"
            bars_html += f"""
<div class="bar-row">
  <span class="bar-label">{CLASS_NAMES[i]}</span>
  <div class="bar-bg"><div class="bar-fill" style="width:{width}"></div></div>
  <span class="bar-pct">{pct:.1f}%</span>
</div>"""
        st.markdown(bars_html + "</div>", unsafe_allow_html=True)

    # Info note
    st.markdown("""
<div class="info-box">
ℹ️ <strong>Note:</strong> CIFAR-10 images are 32×32 px. 
Your image is automatically resized to 32×32 before inference. 
For best results, use images containing a single clear object from one of the 10 supported classes.
</div>
""", unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="info-box">
    📤 Upload an image above to get a prediction. The model can classify:
    airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships, and trucks.
    </div>
    """, unsafe_allow_html=True)

    # Sample grid placeholder
    st.markdown("---")
    st.markdown("#### 🎯 How it works")
    cols = st.columns(3)
    steps = [
        ("1️⃣ Upload", "Choose any JPG/PNG image from your device."),
        ("2️⃣ Preprocess", "Image is resized to 32×32 and normalized."),
        ("3️⃣ Predict", "CNN outputs probabilities for all 10 classes."),
    ]
    for col, (title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"**{title}**")
            st.caption(desc)