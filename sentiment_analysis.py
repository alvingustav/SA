import streamlit as st
import pandas as pd
import re
import nltk
import joblib
import requests
from io import BytesIO
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.svm import SVC

# URL Model dan Vectorizer
MODEL_URL = "https://drive.google.com/uc?id=1P4JCHaYLi6URwEW73Y011XQBVlc55lLs"
VECTORIZER_URL = "https://drive.google.com/uc?id=1tKq520rg80gWryvBKyhxWS1LpzXCZIYS"

# Streamlit App Configuration
st.set_page_config(page_title="Analisis Sentimen Ulasan Film", page_icon="🎬", layout="centered")

# Setup session state
if 'model' not in st.session_state:
    st.session_state['model'] = None

if 'vectorizer' not in st.session_state:
    st.session_state['vectorizer'] = None

# Function to download and load model
@st.cache_resource
def load_model_from_drive(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return joblib.load(BytesIO(response.content))
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    tokens = nltk.word_tokenize(text)  # Tokenization

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)

# Function to predict sentiment
def predict_sentiment(text, model, vectorizer):
    processed_text = preprocess_text(text)

    if not processed_text:
        return "Negatif", 50.0, "Empty text after processing"

    vectorized_text = vectorizer.transform([processed_text])

    try:
        prediction = model.predict(vectorized_text)

        # Confidence score handling for SVM
        if hasattr(model, 'decision_function'):
            confidence = abs(model.decision_function(vectorized_text)[0]) * 100
        else:
            confidence = 70.0  # Default confidence

        # Map prediction output to labels
        sentiment = "Positif" if prediction[0] == 1 else "Negatif"
        return sentiment, confidence, processed_text
    except Exception as e:
        st.error(f"Error saat prediksi: {str(e)}")
        return "Error", 0.0, "N/A"

# Load model and vectorizer if not already loaded
if st.session_state['model'] is None or st.session_state['vectorizer'] is None:
    with st.spinner("🔄 Mengunduh model, harap tunggu..."):
        st.session_state['model'] = load_model_from_drive(MODEL_URL)
        st.session_state['vectorizer'] = load_model_from_drive(VECTORIZER_URL)

# Main UI
st.title("🎬 Analisis Sentimen Ulasan Film")
st.write("Masukkan ulasan film dalam Bahasa Inggris, lalu sistem akan memprediksi apakah sentimennya positif atau negatif.")

# Create tab layout
tab1, tab2 = st.tabs(["Prediksi Sentimen", "Info Aplikasi"])

with tab1:
    user_input = st.text_area("Masukkan ulasan film (dalam Bahasa Inggris):", height=150, 
                              placeholder="Contoh: This movie was fantastic with great acting and an engaging storyline!")

    predict_button = st.button("Prediksi Sentimen", type="primary")

    result_container = st.empty()

    if predict_button:
        if user_input.strip() == "":
            result_container.warning("⚠️ Harap masukkan teks ulasan terlebih dahulu!")
        else:
            if st.session_state['model'] and st.session_state['vectorizer']:
                with st.spinner('Menganalisis sentimen...'):
                    try:
                        model = st.session_state['model']
                        vectorizer = st.session_state['vectorizer']
                        hasil_prediksi, confidence, processed_text = predict_sentiment(user_input, model, vectorizer)

                        with result_container.container():
                            if hasil_prediksi == "Positif":
                                st.success(f"🎯 Prediksi Sentimen: {hasil_prediksi}")
                                emoji = "🎉"
                            else:
                                st.error(f"🎯 Prediksi Sentimen: {hasil_prediksi}")
                                emoji = "😔"

                            st.metric("Tingkat kepercayaan", f"{confidence:.1f}%")
                            st.markdown(f"{emoji} **Ulasan Anda:** {user_input}")

                            with st.expander("Lihat detail pemrosesan teks"):
                                st.markdown("**Tahapan Praproses:**")
                                st.markdown("1. Menghapus tanda baca")
                                st.markdown("2. Mengubah ke huruf kecil")
                                st.markdown("3. Menghapus stopwords")
                                st.markdown("4. Lemmatization")
                                st.markdown("**Hasil setelah praproses:**")
                                st.code(processed_text)

                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat memproses: {str(e)}")
            else:
                result_container.error("Model belum berhasil dimuat. Silakan refresh halaman.")

with tab2:
    st.markdown("### Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini menggunakan model **SVM (Support Vector Machine)** untuk menganalisis sentimen dari ulasan film dalam Bahasa Inggris.

    **Fitur aplikasi:**
    - Prediksi sentimen positif atau negatif
    - Praproses teks otomatis
    - Tingkat kepercayaan prediksi

    **Teknologi yang digunakan:**
    - Streamlit untuk antarmuka pengguna
    - NLTK untuk pemrosesan bahasa alami
    - Scikit-learn untuk model machine learning
    """)

    st.markdown("### Contoh Ulasan")
    
    st.markdown("**Contoh ulasan positif:**")
    positive_examples = [
        "This movie was fantastic! I loved the storyline and the acting was superb.",
        "One of the best films I've seen this year. Great direction and amazing performances."
    ]
    
    for i, example in enumerate(positive_examples):
        if st.button(f"Gunakan contoh positif #{i+1}", key=f"pos_{i}"):
            st.session_state['example_text'] = example
            st.experimental_rerun()

    st.markdown("**Contoh ulasan negatif:**")
    negative_examples = [
        "I was disappointed by the plot and the characters were poorly developed.",
        "The special effects were terrible and the storyline made no sense."
    ]
    
    for i, example in enumerate(negative_examples):
        if st.button(f"Gunakan contoh negatif #{i+1}", key=f"neg_{i}"):
            st.session_state['example_text'] = example
            st.experimental_rerun()

# Handle example text selection
if 'example_text' in st.session_state:
    st.session_state.user_input = st.session_state.example_text
    del st.session_state['example_text']

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: small;'>"
    "Aplikasi Analisis Sentimen dibuat dengan Streamlit"
    "</div>", 
    unsafe_allow_html=True
)
