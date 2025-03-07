import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import pickle
import requests
import io
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Set page config
st.set_page_config(
    page_title="Analisis Sentimen Ulasan Film",
    page_icon="🎬",
    layout="centered"
)

# Progress indicator
progress_placeholder = st.empty()
progress_bar = progress_placeholder.progress(0)

# Function to setup NLTK
@st.cache_resource
def setup_nltk():
    # Set tempat penyimpanan NLTK di lokasi yang dapat diakses oleh Streamlit Cloud
    nltk_data_path = "./.nltk_data"
    if not os.path.exists(nltk_data_path):
        os.makedirs(nltk_data_path)
    
    nltk.data.path.append(nltk_data_path)
    
    # Download required NLTK packages
    packages = ['punkt', 'stopwords', 'wordnet']
    for i, package in enumerate(packages):
        progress_bar.progress((i / len(packages)) * 0.5)  # 50% of progress for NLTK setup
        try:
            nltk.data.find(f'tokenizers/{package}')
        except LookupError:
            nltk.download(package, download_dir=nltk_data_path)
    
    progress_bar.progress(0.5)  # 50% complete
    return "NLTK setup complete"

# Initialize NLTK
setup_status = setup_nltk()

# ========================
# 1️⃣ BUAT MODEL SEDERHANA SEBAGAI FALLBACK
# ========================
@st.cache_resource
def create_fallback_model():
    # Contoh data sederhana untuk melatih model fallback
    simple_data = [
        ("This movie was amazing and I loved it", "positive"),
        ("Great acting and storyline", "positive"),
        ("I enjoyed watching this film", "positive"),
        ("Best movie I've seen all year", "positive"),
        ("Excellent cinematography and direction", "positive"),
        ("The movie was terrible and boring", "negative"),
        ("I didn't like the characters at all", "negative"),
        ("Poor script and bad acting", "negative"),
        ("Waste of time and money", "negative"),
        ("Disappointing plot with too many holes", "negative")
    ]
    
    texts, labels = zip(*simple_data)
    
    # Buat vectorizer dan model
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(texts)
    
    model = LogisticRegression()
    model.fit(X, labels)
    
    progress_bar.progress(0.8)  # 80% complete
    
    return model, vectorizer

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
def predict_sentiment(text, model, vectorizer):
    processed_text = preprocess_text(text)  # Praproses input
    vectorized_text = vectorizer.transform([processed_text])  # Transformasi ke TF-IDF
    prediction = model.predict(vectorized_text)  # Prediksi sentimen
    
    # Get probability scores if the model supports it
    try:
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(vectorized_text)
            confidence = max(proba[0]) * 100
        else:
            confidence = None
    except:
        confidence = None
    
    sentiment = "Positif" if prediction[0] == 'positive' else "Negatif"
    return sentiment, confidence, processed_text

# Load or create model
try:
    # Coba buat model fallback
    fallback_model, fallback_vectorizer = create_fallback_model()
    models_loaded = True
    st.session_state['model'] = fallback_model
    st.session_state['vectorizer'] = fallback_vectorizer
    progress_bar.progress(1.0)  # 100% complete
    progress_placeholder.empty()  # Remove progress bar
except Exception as e:
    st.error(f"Error creating fallback model: {e}")
    models_loaded = False
    progress_placeholder.empty()

# ========================
# 4️⃣ MEMBUAT UI STREAMLIT
# ========================
st.title("🎬 Analisis Sentimen Ulasan Film")
st.write("Masukkan ulasan film dalam Bahasa Inggris di bawah ini, lalu sistem akan memprediksi apakah sentimennya positif atau negatif.")

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
                        model = st.session_state['model']
                        vectorizer = st.session_state['vectorizer']
                        hasil_prediksi, confidence, processed_text = predict_sentiment(user_input, model, vectorizer)
                        
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
    Aplikasi ini menggunakan model machine learning sederhana untuk menganalisis sentimen dari ulasan film dalam Bahasa Inggris.
    
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
    
    examples = {
        "Positif": [
            "This movie was fantastic! I loved the storyline and the acting was superb.",
            "One of the best films I've seen this year. Great direction and amazing performances.",
            "The special effects were amazing and the plot was engaging from start to finish."
        ],
        "Negatif": [
            "I was disappointed by the plot and the characters were poorly developed.",
            "The special effects were terrible and the storyline made no sense.",
            "Waste of time and money. The acting was wooden and the dialogue was terrible."
        ]
    }
    
    for sentiment, texts in examples.items():
        st.markdown(f"**Contoh ulasan {sentiment.lower()}:**")
        for i, text in enumerate(texts):
            if st.button(f"Gunakan contoh {sentiment.lower()} #{i+1}", key=f"{sentiment}_{i}"):
                # Set to session state and rerun to populate the text area
                st.session_state['example_text'] = text
                st.experimental_rerun()
    
    # Check if example text exists in session state
    if 'example_text' in st.session_state:
        # This will be executed after the rerun
        st.session_state['user_input'] = st.session_state['example_text']

# Add footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: small;'>"
    "Aplikasi Analisis Sentimen dibuat dengan Streamlit"
    "</div>", 
    unsafe_allow_html=True
)
