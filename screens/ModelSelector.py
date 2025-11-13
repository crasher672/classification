  # components/ModelSelector.py
import streamlit as st

def show_page(svm_normal, svm_smote, cnn_model):
    st.title("🥥 Coconut Age Classifier")
    st.subheader("Choose which model you want to use")

    st.markdown("""
    - *Without SMOTE* → Original model trained on raw data  
    - *With SMOTE* → Balances training data for fairer classification
    """)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🧠 Use Model Without SMOTE"):
            st.session_state.selected_model = "normal"
            st.session_state.svm_model = svm_normal
            st.session_state.cnn_model = cnn_model
            st.session_state.page = "home_input"
            st.rerun()

    with col2:
        if st.button("🧩 Use Model With SMOTE"):
            st.session_state.selected_model = "smote"
            st.session_state.svm_model = svm_smote
            st.session_state.cnn_model = cnn_model
            st.session_state.page = "home_input"
            st.rerun()