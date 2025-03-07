import streamlit as st
import joblib
import nltk
import os
import requests
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer

# Pastikan NLTK memiliki resource yang dibutuhkan
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# URL Model & Vectorizer dari Google Drive
MODEL_URL = "https://drive.google.com/uc?id=1P4JCHaYLi6URwEW73Y011XQBVlc55lLs"
VECTORIZER_URL = "https://drive.google.com/uc?id=1tKq520rg80gWryvBKyhxWS1LpzXCZIYS"

@st.cache_resource()
def load_model():
    """Fungsi untuk mengunduh dan memuat model SVM dan vectorizer."""
    try:
        # Download model SVM
        response = requests.get(MODEL_URL)
        model = joblib.load(BytesIO(response.content))
        
        # Download vectorizer
        response = requests.get(VECTORIZER_URL)
        vectorizer = joblib.load(BytesIO(response.content))
        
        return model, vectorizer
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None, None

# Load model dan vectorizer
model, vectorizer = load_model()

def preprocess_text(text):
    """Preprocessing teks sebelum diprediksi."""
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(tokens)

def predict_sentiment(text):
    """Memprediksi sentimen teks."""
    if model is None or vectorizer is None:
        return "Error: Model tidak dimuat!"
    
    processed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([processed_text])
    prediction = model.predict(vectorized_text)[0]
    confidence = model.decision_function(vectorized_text)[0]
    
    label = "Positif" if prediction == 1 else "Negatif"
    return label, confidence, processed_text

# Streamlit UI
st.title("🔥 Sentiment Analysis dengan SVM")
user_input = st.text_area("Masukkan teks:")

if st.button("Prediksi Sentimen"):
    if user_input.strip():
        hasil_prediksi, confidence, processed_text = predict_sentiment(user_input)
        st.write(f"**Sentimen:** {hasil_prediksi}")
        st.write(f"**Confidence Score:** {confidence:.4f}")
        st.write(f"**Teks setelah diproses:** {processed_text}")
    else:
        st.warning("Harap masukkan teks sebelum melakukan prediksi!")
