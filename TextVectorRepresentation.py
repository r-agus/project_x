"""
Text Vector Representation Utilities
====================================

This module provides a unified collection of functions for generating text 
representations used in machine-learning experiments. It implements several 
commonly used embedding strategies and the preprocessing tools they rely on.

The module includes the following functionality:

Data Handling
-------------
- **load_data(path)**  
  Loads a dataset from CSV into a Pandas DataFrame.

- **divide_train_val_test(df)**  
  Splits a dataset into train/validation/test subsets following fixed proportions.

- **separate_x_y_vectors(df)**  
  Extracts feature text (tweets) and associated label columns.

Text Preprocessing
------------------
- **preserve_letters(text)**  
  Preserves specific language-sensitive characters (e.g., 'ñ') during unicode 
  normalization.

- **preprocess_text(text)**  
  Performs lowercasing, accent handling, punctuation removal and stopword filtering
  to prepare text for token-based embeddings.

Vectorization Methods
---------------------
- **TF-IDF Representation**: ``vectorRepresentation_TFIDF(xtrain, xval, xtest)``  
  Generates sparse TF-IDF matrices for train, validation and test sets using a 
  consistent vectorizer. Suitable for linear models.

- **BERT Sentence Embeddings**: ``vectorRepresentation_BERT(xtrain, xval, xtest)``  
  Uses *bert-base-multilingual-cased* from HuggingFace Transformers to compute 
  tweet embeddings via mean pooling over token embeddings.

- **Word2Vec Embeddings**: ``vectorRepresentation_Word2Vec(xtrain, xval, xtest)``  
  Trains a Word2Vec model on the training corpus and generates dense vector 
  representations by averaging token embeddings for each tweet. Handles tweets 
  with no valid vocabulary tokens by returning zero vectors.

Design Notes
------------
- All vectorization functions ensure that train/validation/test splits are processed 
  consistently and independently.
- Heavy external models (e.g., BERT) are instantiated inside functions to avoid 
  loading them unnecessarily.
- The module is intentionally self-contained so that other parts of the project can 
  call any representation method interchangeably.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from stopwords import stopwords
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
from init import xtrain
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
import re
import unicodedata
import os

def load_data(file_path: str) -> pd.DataFrame:
    """Loads the dataset from a CSV file.
    
    Main function to load the dataset from a CSV file. It assumes the first row contains headers.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    data = pd.read_csv(file_path, header=0)
    return data


def divide_train_val_test(data: pd.DataFrame, train_size: float = 0.7, val_size: float = 0.15, test_size: float = 0.15):
    '''
    Divides the dataset into training, validation, and test sets.
    Train = 70%, Validation = 15%, Test = 15%
    Args:
        data (pd.DataFrame): The dataset to be divided.
        train_size (float): Proportion of the dataset to include in the training set.
        val_size (float): Proportion of the dataset to include in the validation set.
        test_size (float): Proportion of the dataset to include in the test set.
    '''
    train, temp = train_test_split(data, test_size=(val_size + test_size), random_state=42)
    val, test = train_test_split(temp, test_size=test_size/(val_size + test_size), random_state=42)
    return train, val, test


def separate_x_y_vectors(data: pd.DataFrame):
    '''
    Separates features and labels from the dataset.
    
    Args:
        data (pd.DataFrame): The dataset.
    '''
    X = data['tweet']
    y = data.iloc[:, 1:-1]
    return X, y


def vectorRepresentation_TFIDF(xtrain, xval, xtest):
    '''
    Function to obtain TF-IDF embeddings for tweets in train, validation, and test sets.
    '''
    train_tweets = pd.Series(xtrain).dropna().astype(str).tolist()
    val_tweets = pd.Series(xval).dropna().astype(str).tolist()
    test_tweets = pd.Series(xtest).dropna().astype(str).tolist()
    
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words=list(stopwords),
        ngram_range=(1, 2)
    )

    X_tfidf_train = vectorizer.fit_transform(train_tweets)
    X_tfidf_val = vectorizer.transform(val_tweets)
    X_tfidf_test = vectorizer.transform(test_tweets)
    
    return X_tfidf_train, X_tfidf_val, X_tfidf_test


def vectorRepresentation_BERT(xtrain, xval, xtest):
    """
    Function to obtain BERT embeddings for tweets in train, validation, and test sets.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = BertModel.from_pretrained('bert-base-multilingual-cased')
    
    def get_embeddings(tweets_data):

        """
        Converts a list of tweets into BERT-based sentence embeddings.
        Tokenizes text, runs it through a pretrained multilingual BERT model,
        and returns mean-pooled embeddings for each tweet.
        """

        tweets = pd.Series(tweets_data).dropna().astype(str).tolist()
        inputs = tokenizer(
            tweets,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=16
        )
        with torch.no_grad():
            outputs = model(**inputs)
        token_embeddings = outputs.last_hidden_state
        tweet_embeddings = torch.mean(token_embeddings, dim=1)
        tweet_embeddings = tweet_embeddings.cpu().numpy()
        return tweet_embeddings
    
    train_embeddings = get_embeddings(xtrain)
    val_embeddings = get_embeddings(xval)
    test_embeddings = get_embeddings(xtest)
    
    return train_embeddings, val_embeddings, test_embeddings

def preserve_letters(text: str, letters: list) -> str:
    """Preserves specific letters during text normalization."""
    placeholders = {letter: f"__PLACEHOLDER_{i}__" for i, letter in enumerate(letters)}
    for k, v in placeholders.items():
        text = text.replace(k, v)
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(ch for ch in text if not unicodedata.combining(ch))
    for k, v in placeholders.items():
        text = text.replace(v, k)
    return text


def preprocess_text(text, letters=['ñ','Ñ']):
    """Preprocesses text: preserves letters, lowercases, removes special chars, tokenizes, removes stopwords."""
    text = preserve_letters(text, letters)
    text = text.lower()
    text = re.sub(r"[^a-zA-ZñÑáéíóúüÁÉÍÓÚÜ\s]", "", text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in stopwords]
    return tokens


def vectorRepresentation_Word2Vec(xtrain, xval, xtest):
    '''
    Function to obtain Word2Vec embeddings for tweets in train, validation, and test sets.
    '''
    train_tweets = pd.Series(xtrain).dropna().astype(str).tolist()
    val_tweets = pd.Series(xval).dropna().astype(str).tolist()
    test_tweets = pd.Series(xtest).dropna().astype(str).tolist()
    
    train_tokens = [preprocess_text(tweet) for tweet in train_tweets]
    
    try:
        model = Word2Vec(
            sentences=train_tokens,
            vector_size=100,
            window=5,
            min_count=2,
            workers=4,
            sg=1
        )
    except Exception as e:
        pass
    
    def get_word2vec_embeddings(tweets_data):
        """
        Converts a list of tweets into BERT-based sentence embeddings.
        Tokenizes text, runs it through a pretrained multilingual BERT model,
        and returns mean-pooled embeddings for each tweet.
        """
        
        embeddings = []
        for tweet in tweets_data:
            tokens = preprocess_text(tweet)
            token_vecs = [model.wv[token] for token in tokens if token in model.wv]
            if len(token_vecs) > 0:
                tweet_embedding = np.mean(token_vecs, axis=0)
            else:
                tweet_embedding = np.zeros(model.vector_size)
            embeddings.append(tweet_embedding)
        return np.array(embeddings)
    
    train_embeddings = get_word2vec_embeddings(train_tweets)
    val_embeddings = get_word2vec_embeddings(val_tweets)
    test_embeddings = get_word2vec_embeddings(test_tweets)
    
    return train_embeddings, val_embeddings, test_embeddings

if __name__ == "__main__":

    path = "Datasets/EvaluationData/politicES_phase_2_train_public.csv"
    data = load_data(path)
    data = data.sample(n=30000, random_state=42).reset_index(drop=True)
    train_data, val_data, test_data = divide_train_val_test(data)
    X_train, y_train = separate_x_y_vectors(train_data)
    X_val, y_val = separate_x_y_vectors(val_data)
    X_test, y_test = separate_x_y_vectors(test_data)
    
    # x_tfidf_train, x_tfidf_val, x_tfidf_test = vectorRepresentation_TFIDF(X_train, X_val, X_test)
    # x_BERT_train, x_BERT_val, x_BERT_test = vectorRepresentation_BERT(X_train, X_val, X_test)
    # x_word2vec_train, x_word2vec_val, x_word2vec_test = vectorRepresentation_Word2Vec(X_train, X_val, X_test)
    
    # Create directory if it doesn't exist
    os.makedirs('ProcessedData', exist_ok=True)

    # Save embeddings to files as .npy
    # np.save('ProcessedData/x_tfidf_train_30000.npy', x_tfidf_train.toarray() if hasattr(x_tfidf_train, 'toarray') else x_tfidf_train)
    # np.save('ProcessedData/x_tfidf_val_30000.npy', x_tfidf_val.toarray() if hasattr(x_tfidf_val, 'toarray') else x_tfidf_val)
    # np.save('ProcessedData/x_tfidf_test_30000.npy', x_tfidf_test.toarray() if hasattr(x_tfidf_test, 'toarray') else x_tfidf_test)
    # np.save('ProcessedData/x_BERT_train_30000.npy', x_BERT_train)
    # np.save('ProcessedData/x_BERT_val_30000.npy', x_BERT_val)
    # np.save('ProcessedData/x_BERT_test_30000.npy', x_BERT_test)
    # np.save('ProcessedData/x_word2vec_train_30000.npy', x_word2vec_train)
    # np.save('ProcessedData/x_word2vec_val_30000.npy', x_word2vec_val)
    # np.save('ProcessedData/x_word2vec_test_30000.npy', x_word2vec_test)
    
    # Save labels
    np.save('ProcessedData/y_train_30000.npy', y_train.values if hasattr(y_train, 'values') else y_train)
    np.save('ProcessedData/y_val_30000.npy', y_val.values if hasattr(y_val, 'values') else y_val)
    np.save('ProcessedData/y_test_30000.npy', y_test.values if hasattr(y_test, 'values') else y_test)

    # print ("Shape of TF-IDF train matrix:", x_tfidf_train.shape)
    # print ("Shape of TF-IDF val matrix:  ", x_tfidf_val.shape)
    # print ("Shape of TF-IDF test matrix: ", x_tfidf_test.shape)

    # print ("\nShape of BERT train matrix:", x_BERT_train.shape)
    # print ("Shape of BERT val matrix:  ", x_BERT_val.shape)
    # print ("Shape of BERT test matrix: ", x_BERT_test.shape)

    # print ("\nShape of Word2Vec train matrix:", x_word2vec_train.shape)
    # print ("Shape of Word2Vec val matrix:  ", x_word2vec_val.shape)
    # print ("Shape of Word2Vec test matrix: ", x_word2vec_test.shape)
