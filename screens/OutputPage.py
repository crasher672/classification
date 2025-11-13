# components/OutputPage.py
import streamlit as st

def show_page():
    st.title("🌴 Prediction Result")

    if "predicted_class" not in st.session_state:
        st.warning("No prediction found. Please go back and upload an image.")
        if st.button("⬅️ Go Back"):
            st.session_state.page = "home_input"
            st.rerun()
        return

    st.image(st.session_state.uploaded_image, caption="Coconut Tree Analyzed", width="stretch")
    st.success(f"✅ Predicted Age Category: **{st.session_state.predicted_class}**")
    st.info(f"Estimated Numeric Age: **{st.session_state.estimated_age} years**")

    st.markdown("---")
    if st.button("⬅️ Back to Input"):
        st.session_state.page = "home_input"
        st.rerun()
