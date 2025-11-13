# app.py
import streamlit as st
from screens import ModelSelector, HomeInput, OutputPage
import joblib
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

st.set_page_config(page_title="🌴 Coconut Age Classifier", page_icon="🥥", layout="centered")

# ---------------------------
# Load Models
# ---------------------------
@st.cache_resource
def load_models():
    svm_normal = joblib.load("models/coconut_age_classifier.pkl")
    svm_smote = joblib.load("models/coconut_age_classifier_smote.pkl")
    cnn_model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
    return svm_normal, svm_smote, cnn_model

svm_normal, svm_smote, cnn_model = load_models()

# ---------------------------
# App Navigation
# ---------------------------
if "page" not in st.session_state:
    st.session_state.page = "model_selector"

if st.session_state.page == "model_selector":
    ModelSelector.show_page(svm_normal, svm_smote, cnn_model)
elif st.session_state.page == "home_input":
    HomeInput.show_page()
elif st.session_state.page == "output_page":
    OutputPage.show_page()
