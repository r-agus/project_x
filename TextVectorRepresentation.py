from sklearn.feature_extraction.text import TfidfVectorizer
from stopwords import stopwords
from main import ytrain

tweets = ytrain.iloc[:, -1].dropna().astype(str).tolist()

vectorizer = TfidfVectorizer(
    max_features=5000,     
    stop_words=list(stopwords),  
    ngram_range=(1,2)      
)

X_tfidf = vectorizer.fit_transform(tweets)

print("Shape de la matriz TF-IDF:", X_tfidf)


from transformers import BertTokenizer, BertModel
import torch


tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased')


inputs = tokenizer(
    tweets[:30],
    return_tensors="pt",
    padding=True,        # rellena hasta la misma longitud
    truncation=True,     # corta si son demasiado largos
    max_length=64        # límite de tokens por tweet (ajústalo según necesidad)
)


with torch.no_grad():
    outputs = model(**inputs)

token_embeddings = outputs.last_hidden_state   

tweet_embeddings = torch.mean(token_embeddings, dim=1)  # shape: (n_tweets, 768)

for i, emb in enumerate(tweet_embeddings):
    print(f"Tweet {i}: embedding shape {emb.shape}")
    print(emb[:10])  
