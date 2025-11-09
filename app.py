# 🥥 Coconut Age Classifier Streamlit App
import streamlit as st
import joblib
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
from PIL import Image
import tempfile
import os

# ---------------------------
# 1️⃣ Load Models
# ---------------------------
st.set_page_config(page_title="Coconut Age Classifier", page_icon="🌴", layout="centered")

@st.cache_resource
def load_models():
    svm_model = joblib.load("coconut_age_classifier.pkl")
    cnn_model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
    return svm_model, cnn_model

svm_model, cnn_model = load_models()

# ---------------------------
# 2️⃣ Helper Functions
# ---------------------------
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

sapling_period = 5
avg_leaves_per_year = 13

def estimate_leaf_scars(age):
    leaf_scars = max((age - sapling_period) * avg_leaves_per_year, 0)
    return leaf_scars

def estimate_age(height, leaf_scars, k=1.0, c=0.0, d=5.0):
    if leaf_scars == 0:
        return 0
    return round(k * (height / leaf_scars) + c * np.log(height) + d, 2)

def assumed_age_from_height(height_cm):
    if 300 <= height_cm < 1000:
        return 12  # [5–20]
    elif 1000 <= height_cm < 2200:
        return 30  # [21–40]
    elif 2200 <= height_cm <= 2600:
        return 50  # [41–60]
    else:
        return 20  # fallback

def predict_coconut_age(img_path, height_trunk):
    img_array = preprocess_image(img_path)
    cnn_features = cnn_model.predict(img_array)

    assumed_age = assumed_age_from_height(height_trunk)
    leaf_scars = estimate_leaf_scars(assumed_age)
    est_age_num = assumed_age

    numeric_features = np.array([[height_trunk, leaf_scars, est_age_num]])
    X_combined = np.hstack([cnn_features, numeric_features])

    pred_int = svm_model.predict(X_combined)[0]
    class_map = {0: "[5-20]", 1: "[21-40]", 2: "[41-60]"}
    predicted_category = class_map.get(pred_int, "Unknown")

    return predicted_category, est_age_num

# ---------------------------
# 3️⃣ Streamlit Interface
# ---------------------------
st.title("🌴 Coconut Age Classifier")
st.write("Upload or take a picture of a coconut tree and input its height to predict its age group.")

# Upload image
uploaded_file = st.file_uploader("📸 Upload Coconut Tree Image", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

# Input tree height
height_input = st.number_input("Enter tree height (in centimeters):", min_value=0, step=10)

# Predict button
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    st.image(uploaded_file, caption="Uploaded Coconut Tree", use_container_width=True)

    if st.button("🔍 Predict Age Group"):
        with st.spinner("Analyzing image and predicting age..."):
            pred_class, est_num_age = predict_coconut_age(temp_path, height_input)
        st.success(f"✅ Predicted Age Category: **{pred_class}**")
        st.info(f"Estimated Numeric Age: **{est_num_age} years**")

        # Clean up temporary image file
        os.remove(temp_path)
