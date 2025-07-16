import streamlit as st
import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# ------------------------------
# Page setup
# ------------------------------
st.set_page_config(page_title="Spam Message Classifier", layout="centered")

# ------------------------------
# Load model and vectorizer
# ------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("spam_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# ------------------------------
# Sidebar - User Input
# ------------------------------
st.sidebar.header("Enter your message")

user_input = st.sidebar.text_area("Message", height=150)

# ------------------------------
# Main Panel
# ------------------------------
st.title("📩 Spam Message Classifier")
st.write("""
This application uses a trained machine learning model (Multinomial Naive Bayes)  
to classify text messages as **Spam** or **Not Spam**.
""")

if user_input:
    input_vector = vectorizer.transform([user_input])
    prediction = model.predict(input_vector)[0]
    prediction_proba = model.predict_proba(input_vector)[0]

    st.subheader("Prediction Result")
    if prediction.lower() == 'spam':
        st.error(f"🔴 This message is classified as: **SPAM**")
    else:
        st.success(f"🟢 This message is classified as: **NOT SPAM**")

    st.subheader("Prediction Probability")
    st.write(f"Probability of Spam: **{prediction_proba[1]:.2%}**")
    st.write(f"Probability of Not Spam: **{prediction_proba[0]:.2%}**")
else:
    st.info("👈 Enter a message in the sidebar to get a prediction.")

# ------------------------------
# Sidebar - Help
# ------------------------------
st.sidebar.markdown("""
---
**How to use this app:**
1. Enter a message in the text box.
2. See if it's classified as spam.
3. View prediction probabilities.
""")
