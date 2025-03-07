import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import pickle
import gdown
import os

# Set lokasi penyimpanan NLTK agar tidak bentrok
nltk_data_path = os.path.expanduser("~/.nltk_data")
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

nltk.data.path.append(nltk_data_path)

def download_nltk_package(package):
    try:
        nltk.data.find(f'tokenizers/{package}')
    except LookupError:
        nltk.download(package, download_dir=nltk_data_path)

download_nltk_package('punkt')
download_nltk_package('stopwords')
download_nltk_package('wordnet')

# ========================
# 1️⃣ MEMUAT MODEL DAN VEKTORISASI DARI GOOGLE DRIVE
# ========================
model_url = "https://drive.google.com/uc?id=1P4JCHaYLi6URwEW73Y011XQBVlc55lLs"
vectorizer_url = "https://drive.google.com/uc?id=1tKq520rg80gWryvBKyhxWS1LpzXCZIYS"

model_path = "model.pkl"
vectorizer_path = "vectorizer.pkl"

if not os.path.exists(model_path):
    gdown.download(model_url, model_path, quiet=False)

if not os.path.exists(vectorizer_path):
    gdown.download(vectorizer_url, vectorizer_path, quiet=False)

with open(model_path, 'rb') as model_file:
    best_model = pickle.load(model_file)

with open(vectorizer_path, 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# ========================
# 2️⃣ FUNGSI PRAPROSES TEKS
# ========================
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Hanya huruf dan spasi
    text = text.lower()  # Lowercase
    tokens = nltk.word_tokenize(text)  # Tokenisasi
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

# ========================
# 3️⃣ FUNGSI PREDIKSI SENTIMEN
# ========================
def predict_sentiment(text):
    processed_text = preprocess_text(text)  # Praproses input
    vectorized_text = vectorizer.transform([processed_text])  # Transformasi ke TF-IDF
    prediction = best_model.predict(vectorized_text)  # Prediksi sentimen
    return "Positif" if prediction[0] == 'positive' else "Negatif"

# ========================
# 4️⃣ MEMBUAT UI STREAMLIT
# ========================
st.title("🎬 Analisis Sentimen Ulasan Film")
st.write("Masukkan ulasan film di bawah ini, lalu sistem akan memprediksi apakah sentimennya positif atau negatif.")

user_input = st.text_area("Masukkan ulasan film:", "")

if st.button("Prediksi Sentimen"):
    if user_input.strip() == "":
        st.warning("Harap masukkan teks ulasan terlebih dahulu!")
    else:
        hasil_prediksi = predict_sentiment(user_input)
        st.subheader(f"🎯 Prediksi Sentimen: {hasil_prediksi}")
