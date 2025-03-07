import streamlit as st
import nltk
import joblib
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import urllib.request
import os

# Download tokenizer NLTK jika belum ada
nltk.download('punkt')
nltk.download('stopwords')

# URL model dan vectorizer dari Google Drive
model_url = "https://drive.google.com/uc?id=1P4JCHaYLi6URwEW73Y011XQBVlc55lLs"
vectorizer_url = "https://drive.google.com/uc?id=1tKq520rg80gWryvBKyhxWS1LpzXCZIYS"

# Path untuk menyimpan model
model_path = "svm_model.pkl"
vectorizer_path = "tfidf_vectorizer.pkl"

# Fungsi untuk mengunduh file dari URL
@st.cache_resource
def download_file(url, save_path):
    if not os.path.exists(save_path):
        urllib.request.urlretrieve(url, save_path)

download_file(model_url, model_path)
download_file(vectorizer_url, vectorizer_path)

# Load model dan vectorizer
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Fungsi preprocessing teks
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)  # Hilangkan angka
    text = text.translate(str.maketrans("", "", string.punctuation))  # Hilangkan tanda baca
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Fungsi prediksi sentimen
def predict_sentiment(text):
    processed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([processed_text])
    prediction = model.predict(vectorized_text)[0]
    confidence = max(model.decision_function(vectorized_text)[0])
    return prediction, confidence, processed_text

# Streamlit UI
st.title("Sentiment Analysis with SVM")
user_input = st.text_area("Masukkan teks:")
if st.button("Prediksi Sentimen"):
    if user_input:
        hasil_prediksi, confidence, processed_text = predict_sentiment(user_input)
        st.write(f"Sentimen: {hasil_prediksi}")
        st.write(f"Confidence Score: {confidence:.2f}")
        st.write(f"Teks setelah diproses: {processed_text}")
    else:
        st.warning("Silakan masukkan teks terlebih dahulu!")
