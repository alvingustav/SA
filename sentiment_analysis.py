import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
import gdown
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
url = "https://drive.google.com/file/d/1AzICnuI_WHX_3a7WivGzzFhZcexVTHGZ"
output = "IMDB_Dataset.csv"
gdown.download(url, output, quiet=False)

try:
    data = pd.read_csv(output, encoding="utf-8", engine="python", error_bad_lines=False, warn_bad_lines=True)
except Exception as e:
    print(f"Error saat membaca CSV: {e}")

# Menghapus duplikasi dan nilai kosong
data = data.dropna()

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
# 3️⃣ PELATIHAN MODEL
# ========================
# Praproses data
data['review'] = data['review'].apply(preprocess_text)

# Memisahkan fitur (X) dan label (y)
X = data['review']
y = data['sentiment']

# Split data menjadi train dan test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ekstraksi fitur dengan TF-IDF
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Menangani ketidakseimbangan kelas menggunakan SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_vectorized, y_train)

# Grid Search untuk tuning hyperparameter
param_grid = {'C': [0.1, 1, 10, 100]}
grid_search = GridSearchCV(LinearSVC(max_iter=5000), param_grid, cv=5)
grid_search.fit(X_train_resampled, y_train_resampled)

# Model terbaik
best_model = grid_search.best_estimator_

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
