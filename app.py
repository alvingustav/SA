import streamlit as st
import pandas as pd
import re
import nltk
import pickle
import os
import gdown
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Mengunduh sumber daya NLTK jika belum diunduh
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')

# ========================
# 1️⃣ FUNGSI PRAPROSES TEKS
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
# 2️⃣ LOADING MODEL DAN VECTORIZER
# ========================
@st.cache_resource
def load_model_and_vectorizer():
    # Path untuk model dan vectorizer
    model_path = "model.pkl"
    vectorizer_path = "vectorizer.pkl"
    
    # Cek apakah file model dan vectorizer sudah ada
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        st.info("🔄 Memuat model untuk pertama kali. Hal ini mungkin memerlukan waktu...")
        
        # Download dataset jika belum ada
        dataset_path = "IMDB_Dataset.csv"
        if not os.path.exists(dataset_path):
            # Menggunakan ID file langsung dari URL Google Drive
            file_id = "1AzICnuI_WHX_3a7WivGzzFhZcexVTHGZ"
            gdown.download(f"https://drive.google.com/uc?id={file_id}", dataset_path, quiet=False)
        
        # Load dataset
        data = pd.read_csv(dataset_path, encoding="utf-8")
        
        # Menghapus duplikasi dan nilai kosong
        data = data.drop_duplicates(subset=['review'])
        data = data.dropna()
        
        # Praproses data
        from sklearn.model_selection import train_test_split
        from sklearn.svm import LinearSVC
        from imblearn.over_sampling import SMOTE
        from sklearn.model_selection import GridSearchCV
        
        # Praproses data
        st.info("🔍 Melakukan praproses data...")
        data['review'] = data['review'].apply(preprocess_text)
        
        # Memisahkan fitur (X) dan label (y)
        X = data['review']
        y = data['sentiment']
        
        # Split data menjadi train dan test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Ekstraksi fitur dengan TF-IDF
        st.info("📊 Mengekstrak fitur dengan TF-IDF...")
        vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        X_train_vectorized = vectorizer.fit_transform(X_train)
        
        # Menangani ketidakseimbangan kelas menggunakan SMOTE
        st.info("⚖️ Menangani ketidakseimbangan kelas dengan SMOTE...")
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_vectorized, y_train)
        
        # Grid Search untuk tuning hyperparameter
        st.info("🔧 Melakukan tuning hyperparameter...")
        param_grid = {'C': [0.1, 1, 10, 100]}
        grid_search = GridSearchCV(LinearSVC(max_iter=5000), param_grid, cv=5)
        grid_search.fit(X_train_resampled, y_train_resampled)
        
        # Model terbaik
        best_model = grid_search.best_estimator_
        
        # Simpan model dan vectorizer
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(vectorizer, f)
        
        st.success("✅ Model berhasil dilatih dan disimpan!")
    
    # Load model dan vectorizer
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    
    return model, vectorizer

# ========================
# 3️⃣ PREDIKSI SENTIMEN
# ========================
def predict_sentiment(text, model, vectorizer):
    processed_text = preprocess_text(text)  # Praproses input
    vectorized_text = vectorizer.transform([processed_text])  # Transformasi ke TF-IDF
    prediction = model.predict(vectorized_text)  # Prediksi sentimen
    return "Positif" if prediction[0] == 'positive' else "Negatif"

# ========================
# 4️⃣ UI STREAMLIT
# ========================
def main():
    st.title("🎬 Analisis Sentimen Ulasan Film")
    st.write("Masukkan ulasan film di bawah ini, lalu sistem akan memprediksi apakah sentimennya positif atau negatif.")
    
    # Load model dan vectorizer
    with st.spinner("Memuat model..."):
        model, vectorizer = load_model_and_vectorizer()
    
    # Input dari pengguna
    user_input = st.text_area("Masukkan ulasan film:", "")
    
    # Jika tombol ditekan, lakukan prediksi
    if st.button("Prediksi Sentimen"):
        if user_input.strip() == "":
            st.warning("Harap masukkan teks ulasan terlebih dahulu!")
        else:
            with st.spinner("Menganalisis sentimen..."):
                hasil_prediksi = predict_sentiment(user_input, model, vectorizer)
            
            if hasil_prediksi == "Positif":
                st.success(f"🎯 Prediksi Sentimen: {hasil_prediksi} 👍")
            else:
                st.error(f"🎯 Prediksi Sentimen: {hasil_prediksi} 👎")

if __name__ == "__main__":
    main()
