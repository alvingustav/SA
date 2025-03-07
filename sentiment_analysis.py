import streamlit as st
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import joblib
import os
import gdown

# Mengunduh sumber daya NLTK
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# URL Model dan Vectorizer
MODEL_URL = "https://drive.google.com/uc?id=1gIMlAIZVA4paIw0uNBYOf07NGNOVvjje"
VECTORIZER_URL = "https://drive.google.com/uc?id=1MVLjr5OVI-KWFh2YzIODl1aa5PuDGuAs"

# Mengunduh model dan vectorizer
@st.cache(allow_output_mutation=True)
def load_model_and_vectorizer():
    model_path = "model.pkl"
    vectorizer_path = "vectorizer.pkl"
    
    if not os.path.exists(model_path):
        gdown.download(MODEL_URL, model_path, quiet=False)
    if not os.path.exists(vectorizer_path):
        gdown.download(VECTORIZER_URL, vectorizer_path, quiet=False)
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

# Fungsi untuk praproses teks
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Aplikasi Streamlit
st.title("Analisis Sentimen Film")

# Input teks dari pengguna
user_input = st.text_area("Masukkan ulasan film Anda di sini:")

if user_input:
    # Praproses teks
    processed_text = preprocess_text(user_input)
    
    # Transformasi teks menggunakan vectorizer
    text_vectorized = vectorizer.transform([processed_text])
    
    # Prediksi sentimen
    prediction = model.predict(text_vectorized)
    
    # Menampilkan hasil prediksi
    st.write(f"Prediksi Sentimen: **{prediction[0]}**")

# Menampilkan informasi tambahan
st.sidebar.title("Tentang Aplikasi Ini")
st.sidebar.info(
    "Aplikasi ini menggunakan model Naïve Bayes untuk menganalisis sentimen dari ulasan film. "
    "Masukkan ulasan film Anda di kotak teks dan aplikasi akan memprediksi apakah ulasan tersebut positif atau negatif."
)
