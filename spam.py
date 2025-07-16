import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

data = pd.read_csv(r"C:\Users\moham\OneDrive\Desktop\AMIT Final Project\spam.csv", encoding='ISO-8859-1')
data.drop_duplicates(inplace=True)

mess = data['v2']
cat = data['v1']

mess_train, mess_test, cat_train, cat_test = train_test_split(mess, cat, test_size=0.2, random_state=42)

cv = CountVectorizer(stop_words='english')
features = cv.fit_transform(mess_train)

model = MultinomialNB()
model.fit(features, cat_train)

def predict(message):
    in_message = cv.transform([message])
    result = model.predict(in_message)
    return result[0]

st.set_page_config(page_title="Spam Detector", layout="centered")
st.markdown("<h1 style='text-align: center; color: #2E7D32;'>Spam Message Detector</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center;'>This application uses a machine learning model to classify messages as <strong>Spam</strong> or <strong>Not Spam</strong>.</p>",
    unsafe_allow_html=True
)

input_mess = st.text_input('Enter your message:')

if st.button('Detect'):
    if input_mess.strip() == "":
        st.warning("Please enter a message to analyze.")
    else:
        output = predict(input_mess)
        if output.lower() == "spam":
            st.error("This message is classified as SPAM.")
        else:
            st.success("This message is classified as NOT SPAM.")
