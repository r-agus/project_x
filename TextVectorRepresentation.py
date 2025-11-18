from sklearn.feature_extraction.text import TfidfVectorizer
from stopwords import stopwords
from main import ytrain

tweets = ytrain.iloc[:, -1].dropna().astype(str).tolist()

vectorizer = TfidfVectorizer(
    max_features=5000,     
    stop_words=list(stopwords),  # convertir tu set en lista
    ngram_range=(1,2)      
)

X_tfidf = vectorizer.fit_transform(tweets)

print("Shape de la matriz TF-IDF:", X_tfidf)