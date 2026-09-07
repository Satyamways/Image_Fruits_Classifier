"""Streamlit interface for the trained fruit CNN."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image


IMAGE_SIZE = (128, 128)
MODEL_PATH = Path("models/fruit_cnn.keras")
CLASS_NAMES_PATH = Path("models/class_names.json")

st.set_page_config(page_title="FruitVision | CNN Classifier", page_icon="🍎", layout="centered")
st.markdown(
    """
    <style>
    .main { background: #fffaf5; }
    .block-container { max-width: 860px; padding-top: 3rem; }
    .hero { padding: 2rem; border-radius: 24px; background: linear-gradient(135deg, #7f1d1d, #f97316); color: white; margin-bottom: 1.5rem; }
    .hero h1 { color: white; margin-bottom: .35rem; }
    .hero p { color: #ffedd5; font-size: 1.05rem; margin: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero"><h1>🍎 FruitVision</h1><p>Classify fruit images using a convolutional neural network.</p></div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("AI Intern Project")
    st.write("This demo loads a CNN trained on labeled fruit images.")
    st.divider()
    st.markdown("**Workflow**")
    st.markdown("1. Prepare labeled images\n2. Train the CNN\n3. Upload a new image\n4. Review class probabilities")

if not MODEL_PATH.exists() or not CLASS_NAMES_PATH.exists():
    st.warning("No trained model found yet. Add your dataset and run `python train.py` first.")
    st.code("python train.py --data-dir data --epochs 15", language="bash")
    st.stop()

@st.cache_resource
def load_artifacts():
    import tensorflow as tf
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
    return model, class_names

model, class_names = load_artifacts()
uploaded_file = st.file_uploader("Upload a fruit image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)
    image_array = np.asarray(image.resize(IMAGE_SIZE), dtype=np.float32) / 255.0
    probabilities = model.predict(np.expand_dims(image_array, axis=0), verbose=0)[0]
    best_index = int(np.argmax(probabilities))
    best_class = class_names[best_index]
    confidence = float(probabilities[best_index])

    st.subheader("Prediction")
    st.success(f"Predicted fruit: **{best_class.title()}**")
    st.metric("Confidence", f"{confidence:.1%}")
    st.subheader("Class probabilities")
    scores = {name.title(): float(probabilities[index]) for index, name in enumerate(class_names)}
    st.bar_chart(scores)
else:
    st.info("Upload an image to get a fruit prediction.")

st.divider()
st.caption("This educational model may be inaccurate for unusual lighting, backgrounds, fruit varieties, or classes not included in training.")
