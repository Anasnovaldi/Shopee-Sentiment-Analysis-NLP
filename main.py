# ============================================
# IMPORT LIBRARY
# ============================================
from google_play_scraper import reviews, Sort
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import csv
import os
import re
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory # untuk text preprocessing
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory # untuk text preprocessing
from io import StringIO # untuk pelabelan
import requests # untuk pelabelan
import seaborn as sns # untuk visualisasi
from wordcloud import WordCloud # untuk visualisasi kata kata paling sering muncul
from sklearn.feature_extraction.text import TfidfVectorizer # Fiture ekstraksi
from sklearn.model_selection import train_test_split # Data Spliting
from sklearn.linear_model import LogisticRegression # Training Model
from sklearn.svm import LinearSVC # Training Model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score # Evaluasi Model

# ============================================
# SCRAPING DATASET SHOPEE INDONESIA
# ============================================
path_csv = 'ulasan_aplikasi_shopee.csv'

if os.path.exists(path_csv):
    print(f"File '{path_csv}' sudah ada, langsung load dari CSV (tidak scraping ulang)...")
    app_reviews_df = pd.read_csv(path_csv)
else:
    print("File belum ada, mulai scraping dari Google Play Store...")
    scrapreview, _ = reviews(
        'com.shopee.id',
        lang='id',
        country='id',
        sort=Sort.MOST_RELEVANT,
        count=1000
    )
    app_reviews_df = pd.DataFrame(scrapreview)
    app_reviews_df.to_csv(path_csv, index=False)
    print(f"Scraping selesai, data disimpan ke '{path_csv}'")

# ============================================
# LOADING DATA / CEK DATA
# ============================================
jumlah_ulasan, jumlah_kolom = app_reviews_df.shape
print(f'''\n
Jumlah ulasan: {jumlah_ulasan}, 
Jumlah kolom : {jumlah_kolom}
\n''')
print("\n5 data teratas:")
print(app_reviews_df[["userName", "content"]].head(5))
print("\nInfo DataFrame:")
app_reviews_df.info()

# ============================================
# DATA CLEANING
# ============================================
df = app_reviews_df.dropna()
df = df.drop_duplicates()
panjang_ulasan_bersih = len(df)
print(f"\nSetelah hapus NaN & duplikat: {panjang_ulasan_bersih} ulasan")

# ============================================
# TEXT PREPROCESSING (dengan caching)
# ============================================
path_clean_csv = 'ulasan_shopee_clean.csv'

if os.path.exists(path_clean_csv):
    print(f"File '{path_clean_csv}' sudah ada, langsung load (skip preprocessing)...")
    df = pd.read_csv(path_clean_csv)
    # Kolom 'tokens' tersimpan sbg string '['kata1', 'kata2']' di CSV, konversi balik ke list
    import ast
    df['tokens'] = df['tokens'].apply(ast.literal_eval)
else:
    print("File clean belum ada, mulai preprocessing...")

    # Case Folding
    def case_folding(text):
        return text.lower()
    df['clean_text'] = df['content'].apply(case_folding)

    # Cleaning
    def cleaning(text):
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    df['clean_text'] = df['clean_text'].apply(cleaning)

    # Normalisasi Slang
    slang_dict = {
        'gak': 'tidak', 'ga': 'tidak', 'nggak': 'tidak', 'tdk': 'tidak',
        'bgt': 'banget', 'yg': 'yang', 'dgn': 'dengan', 'utk': 'untuk',
        'aja': 'saja', 'jd': 'jadi', 'krn': 'karena', 'bs': 'bisa',
        'sm': 'sama', 'dr': 'dari', 'udah': 'sudah', 'blm': 'belum',
        'bgs': 'bagus', 'bgus': 'bagus', 'jlk': 'jelek', 'jlek': 'jelek',
        'ska': 'suka', 'ngga': 'tidak', 'mngkin': 'mungkin', 'kren': 'keren',
        'sma': 'sama', 'jdi': 'jadi', 'krna': 'karena', 'bsa': 'bisa',
        'prbaiki': 'perbaiki', 'mntp': 'mantap',
        'ongkir': 'ongkos kirim', 'cod': 'bayar di tempat', 'promo': 'promosi',
        'diskon': 'potongan harga', 'free': 'gratis', 'gratisin': 'gratiskan',
        'parah': 'buruk sekali', 'lemot': 'lambat', 'lelet': 'lambat',
        'error': 'kesalahan', 'nge-lag': 'lambat', 'ngelag': 'lambat',
        'force close': 'tertutup paksa', 'fc': 'tertutup paksa', 'ngebug': 'bermasalah',
        'ribet': 'rumit', 'ribetin': 'merumitkan', 'ribetnya': 'rumitnya',
        'banget': 'sekali', 'bgtt': 'sekali', 'bngt': 'sekali',
        'bener': 'benar', 'emg': 'memang', 'emang': 'memang',
        'knp': 'kenapa', 'kenapa': 'kenapa', 'gmn': 'bagaimana', 'gimana': 'bagaimana',
        'org': 'orang', 'skrg': 'sekarang', 'trs': 'terus', 'tp': 'tapi',
        'wkt': 'waktu', 'bnyk': 'banyak', 'byk': 'banyak', 'dpt': 'dapat',
        'puas': 'puas', 'kecewa': 'kecewa', 'nyesel': 'menyesal',
        'rekomen': 'rekomendasi', 'recommended': 'direkomendasikan',
        'top': 'bagus sekali', 'the best': 'terbaik', 'worst': 'terburuk',
    }
    def normalize_slang(text):
        words = text.split()
        normalized = [slang_dict.get(word, word) for word in words]
        return ' '.join(normalized)
    df['clean_text'] = df['clean_text'].apply(normalize_slang)

    # Tokenisasi
    def tokenize(text):
        return text.split()
    df['tokens'] = df['clean_text'].apply(tokenize)

    # Stopword Removal
    stopword_factory = StopWordRemoverFactory()
    stopword_remover = stopword_factory.create_stop_word_remover()
    def remove_stopwords(text):
        return stopword_remover.remove(text)
    df['clean_text'] = df['clean_text'].apply(remove_stopwords)
    df['tokens'] = df['clean_text'].apply(tokenize)

    # Stemming
    stemmer_factory = StemmerFactory()
    stemmer = stemmer_factory.create_stemmer()
    def stemming(text):
        return stemmer.stem(text)
    df['clean_text'] = df['clean_text'].apply(stemming)
    df['tokens'] = df['clean_text'].apply(tokenize)

    print("\nContoh hasil sebelum vs sesudah preprocessing:")
    print(df[['content', 'clean_text']].head(5))

    df.to_csv(path_clean_csv, index=False)
    print(f"\nData bersih disimpan ke '{path_clean_csv}'")

# ============================================
# PELABELAN (LEXICON BASED)
# ============================================
def load_lexicon(url):
    lexicon = dict()
    response = requests.get(url)
    if response.status_code == 200:
        reader = csv.reader(StringIO(response.text), delimiter=',')
        for row in reader:
            lexicon[row[0]] = int(row[1])
    else:
        print(f"Gagal mengambil data lexicon dari {url}")
    return lexicon

print("\nMengunduh kamus lexicon positif & negatif...")
lexicon_positive = load_lexicon('https://raw.githubusercontent.com/angelmetanosaa/dataset/main/lexicon_positive.csv')
lexicon_negative = load_lexicon('https://raw.githubusercontent.com/angelmetanosaa/dataset/main/lexicon_negative.csv')
print(f"Lexicon positif: {len(lexicon_positive)} kata, Lexicon negatif: {len(lexicon_negative)} kata")

def sentiment_analysis_lexicon_indonesia(text):
    score = 0
    for word in text:  # 'text' harus berupa list token, bukan string
        if word in lexicon_positive:
            score += lexicon_positive[word]
        if word in lexicon_negative:
            score += lexicon_negative[word]

    if score > 0:
        polarity = 'positive'
    elif score < 0:
        polarity = 'negative'
    else:
        polarity = 'neutral'
    return score, polarity

# Pastikan pakai kolom 'tokens' (list token), bukan string
results = df['tokens'].apply(sentiment_analysis_lexicon_indonesia)
results = list(zip(*results))
df['polarity_score'] = results[0]
df['polarity'] = results[1]

print("\nDistribusi label sentimen:")
print(df['polarity'].value_counts())

# Simpan hasil ke CSV
path_labeled_csv = 'ulasan_shopee_labeled.csv'
df.to_csv(path_labeled_csv, index=False)
print(f"\nData berlabel disimpan ke '{path_labeled_csv}'")

# ============================================
# VISUALISASI: DISTRIBUSI SENTIMEN (fixed)
# ============================================
plt.figure(figsize=(6, 4))
sns.countplot(
    data=df, x='polarity', hue='polarity',
    palette={'positive': 'green', 'negative': 'red', 'neutral': 'gray'},
    legend=False
)
plt.title('Distribusi Sentimen Ulasan')
plt.xlabel('Polaritas')
plt.ylabel('Jumlah Ulasan')
plt.show()

# ============================================
# VISUALISASI: WORDCLOUD (fixed: text_akhir -> clean_text)
# ============================================
teks_positif = ' '.join(df[df['polarity'] == 'positive']['clean_text'].dropna())
teks_negatif = ' '.join(df[df['polarity'] == 'negative']['clean_text'].dropna())
teks_netral = ' '.join(df[df['polarity'] == 'neutral']['clean_text'].dropna())

def tampilkan_wordcloud(teks, judul, colormap):
    if teks.strip():
        wc = WordCloud(width=800, height=400, background_color='white', colormap=colormap).generate(teks)
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(judul)
        plt.show()
    else:
        print(f"Tidak ada data untuk: {judul}")

tampilkan_wordcloud(teks_positif, 'WordCloud - Sentimen Positif', 'Greens')
tampilkan_wordcloud(teks_negatif, 'WordCloud - Sentimen Negatif', 'Reds')
tampilkan_wordcloud(teks_netral, 'WordCloud - Sentimen Netral', 'Greys')

# ============================================
# DATA SPLITTING DAN EKSTRAKSI FITUR (TF-IDF)
# ============================================
tfidf = TfidfVectorizer(max_features=5000, min_df=5, max_df=0.8)
X = tfidf.fit_transform(df['clean_text'])   # fit sekali saja, langsung dari teks
y = df['polarity']

features_df = pd.DataFrame(X.toarray(), columns=tfidf.get_feature_names_out())
print("\nContoh hasil ekstraksi fitur TF-IDF:")
print(features_df.head())

print("\nDistribusi kelas sebelum split:")
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nJumlah data train: {X_train.shape[0]}, data test: {X_test.shape[0]}")

# ============================================
# TRAINING MODEL
# ============================================
# Model 1: Logistic Regression
lr = LogisticRegression(max_iter=1000, class_weight='balanced')
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)

# Model 2: SVM
svm = LinearSVC(class_weight='balanced')
svm.fit(X_train, y_train)
pred_svm = svm.predict(X_test)

# ============================================
# EVALUASI MODEL
# ============================================
def evaluasi_model(y_true, y_pred, nama_model):
    print(f"\n{'='*50}")
    print(f"EVALUASI: {nama_model}")
    print(f"{'='*50}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    # Confusion Matrix
    labels = sorted(y_true.unique())
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - {nama_model}')
    plt.xlabel('Prediksi')
    plt.ylabel('Aktual')
    plt.tight_layout()
    plt.show()

    return accuracy_score(y_true, y_pred)

acc_lr = evaluasi_model(y_test, pred_lr, "Logistic Regression")
acc_svm = evaluasi_model(y_test, pred_svm, "SVM (Linear Kernel)")

# ============================================
# PERBANDINGAN FINAL
# ============================================
print(f"\n{'='*50}")
print("PERBANDINGAN AKHIR")
print(f"{'='*50}")
print(f"Logistic Regression Accuracy: {acc_lr:.4f}")
print(f"SVM Accuracy: {acc_svm:.4f}")
print(f"Model terbaik: {'Logistic Regression' if acc_lr > acc_svm else 'SVM'}")