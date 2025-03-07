import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Set page config
st.set_page_config(
    page_title="Analisis Sentimen Ulasan Film",
    page_icon="🎬",
    layout="centered"
)

# Setup state for first run checks
if 'setup_done' not in st.session_state:
    st.session_state['setup_done'] = False
    
if 'model_ready' not in st.session_state:
    st.session_state['model_ready'] = False

# Setup NLTK - removed caching to avoid errors
def setup_nltk():
    # NLTK setup - simplified
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
        
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
        
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    
    return True

# Create simple model function - removed caching
def create_simple_model():
    # Contoh data sederhana untuk melatih model sentiment
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
    
    return model, vectorizer

# Show loading message when app first loads
if not st.session_state['setup_done']:
    setup_container = st.container()
    with setup_container:
        setup_message = st.info("Menyiapkan aplikasi. Mohon tunggu sebentar...")
        
        # Setup NLTK packages
        nltk_status = setup_nltk()
        
        # Create simple model
        try:
            model, vectorizer = create_simple_model()
            st.session_state['model'] = model
            st.session_state['vectorizer'] = vectorizer
            st.session_state['model_ready'] = True
        except Exception as e:
            st.error(f"Terjadi kesalahan saat membuat model: {str(e)}")
        
        st.session_state['setup_done'] = True
        
        # Remove setup message after done
        setup_message.empty()

# Functions for text preprocessing and prediction
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Hanya huruf dan spasi
    text = text.lower()  # Lowercase
    tokens = nltk.word_tokenize(text)  # Tokenisasi
    
    # Get stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Stemming 
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    
    return ' '.join(tokens)

def predict_sentiment(text, model, vectorizer):
    processed_text = preprocess_text(text)  # Praproses input
    
    if not processed_text:  # Handle empty text after processing
        return "Negatif", 50.0, "Empty text after processing"
        
    vectorized_text = vectorizer.transform([processed_text])  # Transformasi ke TF-IDF
    prediction = model.predict(vectorized_text)  # Prediksi sentimen
    
    # Get probability scores
    try:
        proba = model.predict_proba(vectorized_text)
        confidence = max(proba[0]) * 100
    except:
        confidence = 70.0  # Default confidence
    
    sentiment = "Positif" if prediction[0] == 'positive' else "Negatif"
    return sentiment, confidence, processed_text

# ========================
# APLIKASI UTAMA
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
            if st.session_state['model_ready']:
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
                            
                            st.metric("Tingkat kepercayaan", f"{confidence:.1f}%")
                            
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
    
    # Example reviews
    st.markdown("**Contoh ulasan positif:**")
    positive_examples = [
        "This movie was fantastic! I loved the storyline and the acting was superb.",
        "One of the best films I've seen this year. Great direction and amazing performances.",
        "The special effects were amazing and the plot was engaging from start to finish."
    ]
    
    for i, example in enumerate(positive_examples):
        if st.button(f"Gunakan contoh positif #{i+1}", key=f"pos_{i}"):
            st.session_state['example_text'] = example
            st.experimental_rerun()
    
    st.markdown("**Contoh ulasan negatif:**")
    negative_examples = [
        "I was disappointed by the plot and the characters were poorly developed.",
        "The special effects were terrible and the storyline made no sense.",
        "Waste of time and money. The acting was wooden and the dialogue was terrible."
    ]
    
    for i, example in enumerate(negative_examples):
        if st.button(f"Gunakan contoh negatif #{i+1}", key=f"neg_{i}"):
            st.session_state['example_text'] = example
            st.experimental_rerun()

# Handle example text selection
if 'example_text' in st.session_state:
    # Go to first tab
    st.session_state.user_input = st.session_state.example_text
    # Clear example text to avoid infinite rerun
    del st.session_state['example_text']

# Add footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: small;'>"
    "Aplikasi Analisis Sentimen dibuat dengan Streamlit"
    "</div>", 
    unsafe_allow_html=True
)
