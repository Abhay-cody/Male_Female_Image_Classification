import streamlit as st
import numpy as np
from PIL import Image
import joblib

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Male vs Female Image Classifier",
    page_icon="🧑",
    layout="centered"
)

# -------------------------------------------------
# Load Model
# -------------------------------------------------
try:
    model = joblib.load("male_female_model.pkl")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# -------------------------------------------------
# Constants
# -------------------------------------------------
IMG_SIZE = 64

# -------------------------------------------------
# IMPORTANT:
# Change this mapping ONLY if your training labels
# were different.
#
# If:
# Female = 0
# Male   = 1
# (Most common)
# -------------------------------------------------

CLASS_NAMES = {
    0: "Female",
    1: "Male"
}

# -------------------------------------------------
# App Title
# -------------------------------------------------
st.title("🧑 Male vs Female Image Classifier")

st.write(
    "Upload a face image to predict whether the person is **Male** or **Female**."
)

# -------------------------------------------------
# Upload Image
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Read Image
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    resized = image.resize((IMG_SIZE, IMG_SIZE))
    resized = np.array(resized)
    resized = resized.flatten().reshape(1, -1)

    # Prediction
    prediction = model.predict(resized)[0]
    probabilities = model.predict_proba(resized)[0]

    # -------------------------------------------------
    # Display Prediction
    # -------------------------------------------------

    predicted_label = CLASS_NAMES[prediction]

    st.markdown("---")
    st.subheader("Prediction")

    if predicted_label == "Male":
        st.success("🧔 **Prediction: MALE**")
    else:
        st.success("👩 **Prediction: FEMALE**")

    # -------------------------------------------------
    # Confidence Scores
    # -------------------------------------------------

    st.subheader("Prediction Confidence")

    for cls, prob in zip(model.classes_, probabilities):
        st.write(f"**{CLASS_NAMES[cls]}:** {prob*100:.2f}%")
        st.progress(float(prob))

    # -------------------------------------------------
    # Technical Information
    # -------------------------------------------------

    with st.expander("Model Information"):
        st.write("Model Classes:", model.classes_)
        st.write("Predicted Class Index:", prediction)
