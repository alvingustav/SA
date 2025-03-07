import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import pickle  # Tambahkan impor pickle
import os

# Set lokasi penyimpanan NLTK agar tidak bentrok
nltk_data_path = os.path.expanduser("~/.nltk_data")
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

nltk.data.path.append(nltk_data_path)

# Cek apakah paket NLTK sudah ada sebelum mengunduh
def download_nltk_package(package):
    try:
        nltk.data.find(f'tokenizers/{package}')
    except LookupError:
        nltk.download(package, download_dir=nltk_data_path)

download_nltk_package('punkt')
download_nltk_package('stopwords')
download_nltk_package('wordnet')

# ========================
# 1️⃣ MEMBACA DATASET
# ========================
# Bagian ini dapat dihapus jika Anda tidak lagi memerlukan pembacaan dataset

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
# 3️⃣ MEMUAT MODEL DAN VEKTORISASI YANG TELAH DISIMPAN
# ========================
with open('model.pkl', 'rb') as model_file:
    best_model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# ========================
# 4️⃣ FUNGSI PREDIKSI SENTIMEN
# ========================
def predict_sentiment(text):
    processed_text = preprocess_text(text)  # Praproses input
    vectorized_text = vectorizer.transform([processed_text])  # Transformasi ke TF-IDF
    prediction = best_model.predict(vectorized_text)  # Prediksi sentimen
    return "Positif" if prediction[0] == 'positive' else "Negatif"

# ========================
# 5️⃣ MEMBUAT UI STREAMLIT
# ========================
st.title("🎬 Analisis Sentimen Ulasan Film")
st.write("Masukkan ulasan film di bawah ini, lalu sistem akan memprediksi apakah sentimennya positif atau negatif.")

# Input dari pengguna
user_input = st.text_area("Masukkan ulasan film:", "")

# Jika tombol ditekan, lakukan prediksi
if st.button("Prediksi Sentimen"):
    if user_input.strip() == "":
        st.warning("Harap masukkan teks ulasan terlebih dahulu!")
    else:
        hasil_prediksi = predict_sentiment(user_input)
        st.subheader(f"🎯 Prediksi Sentimen: {hasil_prediksi}")
