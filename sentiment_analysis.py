import streamlit as st
import re
import nltk
import time
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import gdown
from PIL import Image

# Judul Aplikasi
st.title("📊 Analisis Sentimen dengan Naive Bayes")
st.write("""
Aplikasi ini memprediksi sentimen dari teks yang Anda masukkan menggunakan model Naive Bayes.
Anda dapat memasukkan kalimat dan melihat prediksi apakah kalimat tersebut memiliki sentimen **Positif** atau **Negatif**.
""")

# Sidebar untuk dokumentasi
st.sidebar.title("📖 Dokumentasi")
st.sidebar.write("""
### 🛠️ Cara Menggunakan:
1. Masukkan teks atau kalimat di kolom input.
2. Klik tombol **Prediksi**.
3. Hasil prediksi akan muncul di bawah, menunjukkan apakah sentimennya **Positif** atau **Negatif**.

### ℹ️ Informasi Model:
- Model yang digunakan adalah **Naive Bayes**.
- Model telah dilatih sebelumnya menggunakan dataset ulasan film IMDB.
- Preprocessing teks meliputi:
  - Menghapus tanda baca.
  - Mengubah teks menjadi lowercase.
  - Menghapus stopwords.
  - Melakukan lemmatization.
""")

# Informasi Dataset
st.sidebar.markdown("---")
st.sidebar.header("📂 Informasi Dataset")
st.sidebar.write("""
Dataset yang digunakan: [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?select=IMDB+Dataset.csv)

**Distribusi Label:**
- Jumlah label:
  - Positif: 24.884
  - Negatif: 24.698
- Persentase:
  - Positif: 50.19%
  - Negatif: 49.81%
""")

# Mengunduh sumber daya NLTK
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Fungsi untuk praproses teks
def preprocess_text(text):
    # Menghapus tanda baca dan mengubah ke lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    # Tokenisasi teks
    tokens = nltk.word_tokenize(text)
    # Menghapus stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Mengunduh model dan vectorizer dari Google Drive
@st.cache
def load_model_and_vectorizer():
    MODEL_URL = "https://drive.google.com/uc?id=1gIMlAIZVA4paIw0uNBYOf07NGNOVvjje"
    VECTORIZER_URL = "https://drive.google.com/uc?id=16fDDLyhz3V2JeTU0Q0Ovul7oFHUyFkui"
    
    # Mengunduh file
    gdown.download(MODEL_URL, "model.pkl", quiet=True)
    gdown.download(VECTORIZER_URL, "vectorizer.pkl", quiet=True)
    
    # Memuat model dan vectorizer
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

# Memuat model dan vectorizer
model, vectorizer = load_model_and_vectorizer()

# Input teks dari pengguna
st.header("✍️ Masukkan Teks untuk Prediksi")
user_input = st.text_area("Masukkan teks di sini:", "This movie was fantastic! The acting was top-notch and the story was very engaging.", height=150)

# Tombol untuk prediksi
if st.button("🚀 Prediksi"):
    with st.spinner("Sedang memproses..."):  # Animasi loading
        time.sleep(1)  # Simulasi proses yang memakan waktu
        # Preprocessing teks input
        processed_text = preprocess_text(user_input)
        # Transformasi teks menggunakan vectorizer
        try:
            text_vectorized = vectorizer.transform([processed_text])
            # Prediksi sentimen
            prediction = model.predict(text_vectorized)[0]
            
            # Menampilkan hasil prediksi
            st.subheader("🎯 Hasil Prediksi")
            if prediction == "positive":
                st.success("✅ Sentimen: **Positif** 😊")
            else:
                st.error("❌ Sentimen: **Negatif** 😠")

        except Exception as e:
            st.error(f"⚠️ Terjadi error saat memproses teks: {e}")

# Visualisasi WordCloud dari Dataset
st.markdown("---")
st.header("📊 WordCloud dari Dataset")
st.write("Berikut adalah visualisasi WordCloud dari dataset yang digunakan untuk melatih model.")

# Mengunduh dan menampilkan WordCloud
WORDCLOUD_URL = "https://drive.google.com/uc?id=1gkeQ6cj87zI7WsU8DF09XuFFSt9MY-oW"
@st.cache
def load_wordcloud():
    gdown.download(WORDCLOUD_URL, "wordcloud.png", quiet=True)
    return Image.open("wordcloud.png")

wordcloud_image = load_wordcloud()
st.image(wordcloud_image, caption="WordCloud dari Dataset IMDB", use_column_width=True)

# Tambahkan footer
st.markdown("---")
st.write("""
### 🛠️ Tentang Aplikasi:
- Dibuat dengan **Streamlit**.
- Menggunakan model **Naive Bayes** untuk analisis sentimen.
- Dataset: [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?select=IMDB+Dataset.csv).
- Kode sumber tersedia di [GitHub](https://github.com).
""")
