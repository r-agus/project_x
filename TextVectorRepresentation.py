from sklearn.feature_extraction.text import TfidfVectorizer
from stopwords import stopwords
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
import torch


ytrain = pd.read_csv('./Datasets/PractiseData/development.csv', header=0)

def vectorRepresentation_TFIDF(ytrain):
    '''
    Function to obtain TF-IDF embeddings for tweets in any ytrain.
    '''
    tweets = ytrain.iloc[:, -1].dropna().astype(str).tolist()
    vectorizer = TfidfVectorizer(
        max_features=5000,     
        stop_words=list(stopwords),  
        ngram_range=(1,2)      
    )

    X_tfidf = vectorizer.fit_transform(tweets)

    return X_tfidf, vectorizer

def vectorRepresentation_BERT(ytrain):
    '''
    Function to obtain BERT embeddings for tweets in any ytrain.
    '''
    # Obtain tweets
    tweets = ytrain.iloc[:, -1].dropna().astype(str).tolist()
    
    # Load BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased') # BERT tokenizer
    model = BertModel.from_pretrained('bert-base-multilingual-cased') # BERT model

    # Tokenize and encode tweets
    inputs = tokenizer(
        tweets[:20],
        return_tensors="pt",
        padding=True,   
        truncation=True,
        max_length=64
    )
    
    # # Debugging example for BERT tokenization
    # example = "El gobierno del PSOE está trabajando en economía."
    # tokens_example = tokenizer.tokenize(example)
    # print(tokens_example)
    
    
    # Get BERT embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the token embeddings from the last hidden state
    token_embeddings = outputs.last_hidden_state 
    tweet_embeddings = torch.mean(token_embeddings, dim=1)
    
    # For debugging
    # for i, emb in enumerate(tweet_embeddings):
    #     print(f"Tweet {i}: embedding shape {emb.shape}")
    #     print(emb[:10])  
    
    return tweet_embeddings

x_tfidf, vectorizer = vectorRepresentation_TFIDF(ytrain)
x_BERT = vectorRepresentation_BERT(ytrain)

print("Shape of TF-IDF matrix:", x_tfidf.shape)
print("Shape of BERT matrix:  ", x_BERT.shape)
