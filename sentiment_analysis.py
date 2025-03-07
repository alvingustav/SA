import streamlit as st
import pandas as pd
import re
import nltk
import joblib
import requests
from io import BytesIO
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')

# URL Model dan Vectorizer
MODEL_URL = "https://drive.google.com/uc?id=1gIMlAIZVA4paIw0uNBYOf07NGNOVvjje"
VECTORIZER_URL = "https://drive.google.com/uc?id=1MVLjr5OVI-KWFh2YzIODl1aa5PuDGuAs"

# Function to download and load model
@st.cache_resource
def load_model_from_drive(url):
    response = requests.get(url)
    response.raise_for_status()
    return joblib.load(BytesIO(response.content))

# Load model & vectorizer
model = load_model_from_drive(MODEL_URL)
vectorizer = load_model_from_drive(VECTORIZER_URL)

# Fungsi untuk praproses teks
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Streamlit UI
st.title("🎬 Analisis Sentimen Ulasan Film")

# Upload CSV File
uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("📊 **5 Data Pertama:**")
    st.dataframe(data.head())
    
    # Cek jika kolom 'review' ada
    if 'review' not in data.columns or 'sentiment' not in data.columns:
        st.error("❌ Dataset harus memiliki kolom 'review' dan 'sentiment'!")
    else:
        # Pratinjau jumlah label
        label_counts = data['sentiment'].value_counts()
        st.write("📊 **Distribusi Label:**", label_counts)
        
        # Visualisasi Bar Chart
        fig, ax = plt.subplots()
        sns.barplot(x=label_counts.index, y=label_counts.values, palette='viridis', ax=ax)
        plt.xlabel('Sentiment')
        plt.ylabel('Jumlah')
        plt.title('Distribusi Label Sentiment')
        st.pyplot(fig)
        
        # Preraproses teks
        data['clean_review'] = data['review'].apply(preprocess_text)
        
        # Visualisasi WordCloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(data['clean_review']))
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(fig)
        
        # Prediksi
        X_vectorized = vectorizer.transform(data['clean_review'])
        y_pred = model.predict(X_vectorized)
        data['predicted_sentiment'] = y_pred
        
        # Evaluasi Model
        st.write("📝 **Laporan Klasifikasi:**")
        st.text(classification_report(data['sentiment'], data['predicted_sentiment']))
        
        # Confusion Matrix
        cm = confusion_matrix(data['sentiment'], data['predicted_sentiment'])
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        st.pyplot(fig)
        
        # Download hasil prediksi
        st.write("📥 **Download Hasil Prediksi**")
        output_csv = data[['review', 'sentiment', 'predicted_sentiment']].to_csv(index=False)
        st.download_button(label="Unduh CSV", data=output_csv, file_name="hasil_prediksi.csv", mime="text/csv")

# Prediksi Ulasan Manual
st.write("### 🎭 Prediksi Sentimen Ulasan Tunggal")
user_input = st.text_area("Masukkan ulasan film:", "The movie was amazing, I loved it!")
if st.button("Prediksi Sentimen"):
    cleaned_text = preprocess_text(user_input)
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)[0]
    sentiment_label = "Positif" if prediction == 1 else "Negatif"
    st.write(f"🎯 **Hasil Prediksi:** {sentiment_label}")
