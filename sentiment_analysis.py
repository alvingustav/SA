import streamlit as st
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import gdown
import joblib

# Mengunduh sumber daya NLTK
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Mengunduh model dan vectorizer dari Google Drive
MODEL_URL = "https://drive.google.com/uc?id=1gIMlAIZVA4paIw0uNBYOf07NGNOVvjje"
VECTORIZER_URL = "https://drive.google.com/uc?id=1MVLjr5OVI-KWFh2YzIODl1aa5PuDGuAs"

@st.cache
def download_file_from_google_drive(url, output):
    gdown.download(url, output, quiet=False)

download_file_from_google_drive(MODEL_URL, "model.pkl")
download_file_from_google_drive(VECTORIZER_URL, "vectorizer.pkl")

# Memuat model dan vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Membaca dataset
data_url = "https://raw.githubusercontent.com/username/repo/main/IMDB_Dataset.csv"  # Ganti dengan URL dataset Anda di GitHub
data = pd.read_csv(data_url)

# Menghapus duplikat dan nilai null
data = data.drop_duplicates(subset=['review']).dropna()

# Fungsi untuk praproses teks
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Melakukan praproses pada kolom 'review'
data['review'] = data['review'].apply(preprocess_text)

# Menampilkan 5 data awal
st.write("5 Data Awal:")
st.write(data.head())

# Menampilkan jumlah label positif dan negatif
label_counts = data['sentiment'].value_counts()
st.write("Jumlah label:\n", label_counts)
st.write("Persentase:\n", label_counts / len(data) * 100)

# Visualisasi wordcloud
st.write("WordCloud:")
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(data['review']))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
st.pyplot(plt)

# Visualisasi label dengan bar chart
st.write("Distribusi Label Sentiment:")
plt.figure(figsize=(6, 4))
sns.barplot(x=label_counts.index, y=label_counts.values, palette='viridis')
plt.xlabel('Sentiment')
plt.ylabel('Jumlah')
plt.title('Distribusi Label Sentiment')
st.pyplot(plt)

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(data['review'], data['sentiment'], test_size=0.2, random_state=42)
st.write("Jumlah data latih:", len(X_train))
st.write("Jumlah data uji:", len(X_test))

# TF-IDF vektorisasi
X_train_vectorized = vectorizer.transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Prediksi dan evaluasi
y_pred = model.predict(X_test_vectorized)
st.write("Classification Report:")
st.write(classification_report(y_test, y_pred))

# Confusion matrix
st.write("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
st.pyplot(plt)

# Prediksi data uji baru
st.write("Prediksi untuk Data Uji Baru:")
sample_text = st.text_area("Masukkan teks untuk prediksi:", "This movie was fantastic! The acting was top-notch and the story was very engaging.")
sample_vectorized = vectorizer.transform([sample_text])
predictions = model.predict(sample_vectorized)
st.write("Prediksi:", predictions[0])
