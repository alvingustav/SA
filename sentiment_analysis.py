import streamlit as st
import pandas as pd
import re
import nltk
import os
import gdown
import joblib
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Set page config
st.set_page_config(page_title="Analisis Sentimen Ulasan Film", page_icon="🎬", layout="centered")

# Setup state
if 'model_ready' not in st.session_state:
    st.session_state['model_ready'] = False

# URL model dan vectorizer
model_url = "https://drive.google.com/uc?id=1P4JCHaYLi6URwEW73Y011XQBVlc55lLs"
vectorizer_url = "https://drive.google.com/uc?id=1tKq520rg80gWryvBKyhxWS1LpzXCZIYS"
model_path = "model.pkl"
vectorizer_path = "vectorizer.pkl"

def download_file(url, output):
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)

def load_model():
    try:
        download_file(model_url, model_path)
        download_file(vectorizer_url, vectorizer_path)
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer
    except Exception as e:
        st.error(f"Gagal memuat model: {str(e)}")
        return None, None

def setup_nltk():
    resources = ['punkt', 'stopwords', 'wordnet']
    for resource in resources:
        try:
            nltk.data.find(f'corpora/{resource}')
        except LookupError:
            nltk.download(resource)
setup_nltk()

# Memuat model jika belum tersedia
if not st.session_state['model_ready']:
    model, vectorizer = load_model()
    if model and vectorizer:
        st.session_state['model'] = model
        st.session_state['vectorizer'] = vectorizer
        st.session_state['model_ready'] = True

# Fungsi praproses teks
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    try:
        tokens = nltk.word_tokenize(text)
    except:
        tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

def predict_sentiment(text):
    processed_text = preprocess_text(text)
    vectorized_text = st.session_state['vectorizer'].transform([processed_text])
    prediction = st.session_state['model'].predict(vectorized_text)
    proba = st.session_state['model'].predict_proba(vectorized_text)
    confidence = max(proba[0]) * 100 if proba is not None else 70.0
    sentiment = "Positif" if prediction[0] == 'positive' else "Negatif"
    return sentiment, confidence, processed_text

# UI Streamlit
st.title("🎬 Analisis Sentimen Ulasan Film")
st.write("Masukkan ulasan film dalam Bahasa Inggris, lalu sistem akan memprediksi sentimennya.")
user_input = st.text_area("Masukkan ulasan film:", height=150, placeholder="Contoh: This movie was fantastic!")
predict_button = st.button("Prediksi Sentimen")

if predict_button:
    if not user_input.strip():
        st.warning("⚠️ Harap masukkan teks ulasan terlebih dahulu!")
    elif st.session_state['model_ready']:
        with st.spinner('Menganalisis sentimen...'):
            hasil_prediksi, confidence, processed_text = predict_sentiment(user_input)
            st.success(f"🎯 Prediksi Sentimen: {hasil_prediksi}")
            st.metric("Tingkat kepercayaan", f"{confidence:.1f}%")
            with st.expander("Lihat detail pemrosesan teks"):
                st.code(processed_text)
    else:
        st.error("Model belum siap, coba refresh halaman.")
