import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import pickle
import gdown
import os

# Set page config
st.set_page_config(
    page_title="Analisis Sentimen Ulasan Film",
    page_icon="🎬",
    layout="centered"  # Changed to centered for better mobile experience
)

# Function to create directory and setup NLTK
@st.cache_resource
def setup_nltk():
    # Set tempat penyimpanan NLTK di lokasi yang dapat diakses oleh Streamlit Cloud
    nltk_data_path = "./.nltk_data"
    if not os.path.exists(nltk_data_path):
        os.makedirs(nltk_data_path)
    
    nltk.data.path.append(nltk_data_path)
    
    # Download required NLTK packages
    for package in ['punkt', 'stopwords', 'wordnet']:
        try:
            nltk.data.find(f'tokenizers/{package}')
        except LookupError:
            with st.spinner(f'Downloading NLTK package: {package}...'):
                nltk.download(package, download_dir=nltk_data_path)
    
    return "NLTK setup complete"

# Initialize NLTK
with st.spinner('Menyiapkan NLTK resources...'):
    setup_status = setup_nltk()

# Download model dan vectorizer function
@st.cache_resource
def load_models():
    model_path = "./model.pkl"
    vectorizer_path = "./vectorizer.pkl"
    
    # Show download status
    with st.spinner('Mengunduh model dari Google Drive...'):
        # Gunakan URL langsung dari Google Drive dengan gdown
        model_url = "https://drive.google.com/uc?id=1P4JCHaYLi6URwEW73Y011XQBVlc55lLs"
        vectorizer_url = "https://drive.google.com/uc?id=1tKq520rg80gWryvBKyhxWS1LpzXCZIYS"
        
        if not os.path.exists(model_path):
            gdown.download(model_url, model_path, quiet=False)
        
        if not os.path.exists(vectorizer_path):
            gdown.download(vectorizer_url, vectorizer_path, quiet=False)
    
    # Load the models
    try:
        with open(model_path, 'rb') as model_file:
            best_model = pickle.load(model_file)
        
        with open(vectorizer_path, 'rb') as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)
        
        return best_model, vectorizer
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

# Show loading indicator while loading models
with st.spinner('Memuat model...'):
    best_model, vectorizer = load_models()

# Check if models loaded successfully
models_loaded = best_model is not None and vectorizer is not None

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
    
    # Get probability scores if the model supports it
    try:
        if hasattr(best_model, 'predict_proba'):
            proba = best_model.predict_proba(vectorized_text)
            confidence = max(proba[0]) * 100
        else:
            confidence = None
    except:
        confidence = None
    
    sentiment = "Positif" if prediction[0] == 'positive' else "Negatif"
    return sentiment, confidence, processed_text

# ========================
# 4️⃣ MEMBUAT UI STREAMLIT
# ========================
st.title("🎬 Analisis Sentimen Ulasan Film")
st.write("Masukkan ulasan film di bawah ini, lalu sistem akan memprediksi apakah sentimennya positif atau negatif.")

# Create container for status messages
status_container = st.empty()

if not models_loaded:
    status_container.warning("⚠️ Model tidak berhasil dimuat. Harap refresh halaman.")

# Create tab layout
tab1, tab2 = st.tabs(["Prediksi Sentimen", "Info Aplikasi"])

with tab1:
    # Input area
    user_input = st.text_area("Masukkan ulasan film (dalam Bahasa Inggris):", height=150, 
                              placeholder="Contoh: This movie was fantastic with great acting and an engaging storyline!")
    
    # Prediction button with custom styling
    predict_button = st.button("Prediksi Sentimen", type="primary", use_container_width=True)
    
    # Results area (hidden until prediction is made)
    result_container = st.empty()
    
    # Run prediction when button is clicked
    if predict_button:
        if user_input.strip() == "":
            result_container.warning("⚠️ Harap masukkan teks ulasan terlebih dahulu!")
        else:
            if models_loaded:
                with st.spinner('Menganalisis sentimen...'):
                    try:
                        hasil_prediksi, confidence, processed_text = predict_sentiment(user_input)
                        
                        # Display the result in a nice container
                        with result_container.container():
                            if hasil_prediksi == "Positif":
                                st.success(f"🎯 Prediksi Sentimen: {hasil_prediksi}")
                                emoji = "🎉"
                            else:
                                st.error(f"🎯 Prediksi Sentimen: {hasil_prediksi}")
                                emoji = "😔"
                            
                            if confidence is not None:
                                st.metric("Tingkat kepercayaan", f"{confidence:.2f}%")
                            
                            st.markdown(f"{emoji} **Ulasan Anda:** {user_input}")
                            
                            # Show processed text for educational purposes
                            with st.expander("Lihat detail pemrosesan teks"):
                                st.markdown("**Tahapan Praproses:**")
                                st.markdown("1. Menghapus tanda baca")
                                st.markdown("2. Mengubah ke huruf kecil")
                                st.markdown("3. Menghapus stopwords (kata umum)")
                                st.markdown("4. Lemmatisasi (mengubah ke bentuk dasar)")
                                st.markdown("5. Stemming (mengekstrak akar kata)")
                                
                                st.markdown("**Hasil setelah praproses:**")
                                st.code(processed_text)
                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat memproses: {str(e)}")
                        st.info("Coba refresh halaman atau gunakan teks yang berbeda.")
            else:
                result_container.error("Model belum berhasil dimuat. Silakan refresh halaman.")

with tab2:
    st.markdown("### Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini menggunakan model machine learning untuk menganalisis sentimen dari ulasan film dalam Bahasa Inggris.
    
    **Fitur aplikasi:**
    - Analisis sentimen positif/negatif
    - Praproses teks otomatis
    - Tingkat kepercayaan prediksi
    
    **Teknologi yang digunakan:**
    - Streamlit untuk antarmuka pengguna
    - NLTK untuk pemrosesan bahasa alami
    - Scikit-learn untuk model machine learning
    
    **Catatan:** Aplikasi ini berjalan paling baik dengan ulasan dalam Bahasa Inggris.
    """)
    
    st.markdown("### Contoh Ulasan")
    st.markdown("""
    **Contoh ulasan positif:**
    - "This movie was fantastic! I loved the storyline and the acting was superb."
    - "One of the best films I've seen this year. Great direction and amazing performances."
    
    **Contoh ulasan negatif:**
    - "I was disappointed by the plot and the characters were poorly developed."
    - "The special effects were terrible and the storyline made no sense."
    """)

# Add footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: small;'>"
    "Aplikasi Analisis Sentimen dibuat dengan Streamlit"
    "</div>", 
    unsafe_allow_html=True
)
