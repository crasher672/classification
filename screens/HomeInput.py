# components/HomeInput.py
import streamlit as st
import tempfile
import os
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from PIL import Image

# ---------------------------
# Helper functions
# ---------------------------
sapling_period = 5
avg_leaves_per_year = 13

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def compute_leaf_scars(height_cm):
    """
    Simple estimation of leaf scars from height.
    """
    assumed_age = max(height_cm / 30, sapling_period)
    leaf_scars = max((assumed_age - sapling_period) * avg_leaves_per_year, 0)
    return leaf_scars, assumed_age

# ---------------------------
# Page UI
# ---------------------------
def show_page():
    # Back button
    top_left, top_right = st.columns([9, 1])
    with top_right:
        if st.button("⬅️ Models"):
            st.session_state.page = "model_selector"
            st.rerun()

    st.title("🌴 Coconut Age Classifier")
    st.write("Upload an image of a coconut tree and input its height to predict its age category.")

    st.markdown("""
    **User Guide in Estimating the Tree Height:**
    - 300 to 1,000 cm → [5–20 years]  
    - 1,001 to 2,200 cm → [21–40 years]  
    - 2,201 to 2,600 cm → [41–60 years]
    """)

    uploaded_file = st.file_uploader("📸 Upload Coconut Tree Image", type=["jpg", "jpeg", "png"])
    height_cm = st.number_input("🌿 Enter tree height (cm):", min_value=0, step=10)

    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Coconut Tree", use_container_width=True)

        if st.button("🔍 Predict Age Group"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_path = temp_file.name

            with st.spinner("Analyzing image and predicting age..."):
                # Load models from session state
                svm_model = st.session_state.svm_model
                cnn_model = st.session_state.cnn_model

                # --- Preprocess image and extract features ---
                img_array = preprocess_image(temp_path)
                cnn_features = cnn_model.predict(img_array, verbose=0)
                
                # --- Compute numeric features ---
                leaf_scars, est_age_guess = compute_leaf_scars(height_cm)
                numeric_features = np.array([[height_cm, leaf_scars, est_age_guess]])

                # --- Combine CNN + numeric features ---
                X_combined = np.hstack([cnn_features.reshape(1, -1), numeric_features])

                # --- Predict age category ---
                predicted_class_int = svm_model.predict(X_combined)[0]
                category_map = {0: "[5-20]", 1: "[21-40]", 2: "[41-60]"}
                predicted_class = category_map.get(predicted_class_int, "Unknown")

                # --- Estimated numeric age ---
                estimated_age = round(est_age_guess, 2)

            os.remove(temp_path)

            # Store results in session state
            st.session_state.predicted_class = predicted_class
            st.session_state.estimated_age = estimated_age
            st.session_state.uploaded_image = uploaded_file
            st.session_state.page = "output_page"
            st.rerun()
