from sklearn.feature_extraction.text import TfidfVectorizer
from stopwords import stopwords
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
from init import xtrain
from sentence_transformers import SentenceTransformer

def vectorRepresentation_TFIDF(xtrain):
    '''
    Function to obtain TF-IDF embeddings for tweets in any ytrain.
    '''
    tweets = pd.Series(xtrain).dropna().astype(str).tolist()
    vectorizer = TfidfVectorizer(
        max_features=5000,     
        stop_words=list(stopwords),  
        ngram_range=(1,2)      
    )

    X_tfidf = vectorizer.fit_transform(tweets)

    return X_tfidf, vectorizer

def vectorRepresentation_BERT(xtrain):
    '''
    Function to obtain BERT embeddings for tweets in any xtrain.
    '''
    # Obtain tweets
    tweets = pd.Series(xtrain).dropna().astype(str).tolist()
    
    # Load BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased') # BERT tokenizer
    model = BertModel.from_pretrained('bert-base-multilingual-cased') # BERT model

    # Tokenize and encode tweets
    inputs = tokenizer(
        tweets,
        return_tensors="pt",
        padding=True,   
        truncation=True,
        max_length=16
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

x_tfidf, vectorizer = vectorRepresentation_TFIDF(xtrain)
x_BERT = vectorRepresentation_BERT(xtrain[:20])

print("Shape of TF-IDF matrix:", x_tfidf.shape)
print("Shape of BERT matrix:  ", x_BERT.shape)


def vectorRepresentation_MiniLM(xtrain):
    """
    Function to obtain BERT embeddings (MiniLM model) for tweets in any xtrain.
    """
    tweets = pd.Series(xtrain).dropna().astype(str).tolist()

    model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

    # (output: a numpy array with shape [num_tweets, 384])
    embeddings = model.encode(tweets, show_progress_bar=True)
    
    return embeddings


x_MiniLM = vectorRepresentation_MiniLM(xtrain)
print("Shape of BERT matrix:  ", x_MiniLM.shape)
