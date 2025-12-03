from sklearn.feature_extraction.text import TfidfVectorizer
from stopwords import stopwords
from main import ytrain
from datasets import Dataset
from transformers import BertTokenizer, BertModel, pipeline, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch

ytrain = ytrain[:30]

tweets = ytrain.iloc[:, -1].dropna().astype(str).tolist()

vectorizerTfidf = TfidfVectorizer(
    max_features=5000,     
    stop_words=list(stopwords),  
    ngram_range=(1,2)      
)

X_tfidf = vectorizerTfidf.fit_transform(tweets)

print("Shape de la matriz TF-IDF:", X_tfidf)

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased')


inputs = tokenizer(
    tweets,
    return_tensors="pt",
    padding=True,        # rellena hasta la misma longitud
    truncation=True,     # corta si son demasiado largos
    max_length=140        # token limit per tweet (a tweet can have 280 caracters and a token is 3-4 caracters, just in case we use 140)
)

with torch.no_grad():
    outputs = model(**inputs)

token_embeddings = outputs.last_hidden_state   

tweet_embeddings = torch.mean(token_embeddings, dim=1)  # shape: (n_tweets, 768)

for i, emb in enumerate(tweet_embeddings):
    print(f"Tweet {i}: embedding shape {emb.shape}")
    print(emb[:10])  

print(token_embeddings[:30])
print(tweet_embeddings[:30])

tweet_embeddings = torch.mean(token_embeddings, dim=1)