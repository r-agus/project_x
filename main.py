# training
# A neural network implemented in PyTorch.
# At least one other Scikit-learn algorithm (for example, K-NN, SVM, Random Forest, Logistic Regression, etc.).

"""Main module for data loading, exploration, and visualization.

Provides functions to load data, print data information, analyze class distribution,
preserve specific letters during text normalization, and generate word clouds.
"""

__all__ = ['load_data', 'print_data_info', 'analyze_class_distribution', 'preserve_letters', 'generate_wordcloud']

import pandas as pd
import numpy as np
from sklearn import neighbors
from unidecode import unidecode
import unicodedata
import re
from stopwords import stopwords
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords as sw
from init import xtrain, ytrain, xtest, ytest, xvalidation, yvalidation
from stopwords import stopwords



def load_data(file_path: str) -> pd.DataFrame:
    """Loads the dataset from a CSV file.
    
    Main funtion to load the dataset from a CSV file. It assumes the first row contains headers.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    data = pd.read_csv(file_path, header=0)
    return data

# traindata = load_data('Datasets/EvaluationData/politicES_phase_2_train_public.csv')
# ytrain = traindata.iloc[:, :]

def print_data_info(ytrain: pd.DataFrame):
    """
    Print structural information about the training dataframe.

    Args:
        ytrain (pd.DataFrame): Training split with user metadata and tweets.

    Returns:
        None
    """
    # Shape of the training data
    print("Training data shape:", ytrain.shape)

    # Print the headers of the dataset
    print("Headers:", ytrain.columns.tolist())

    # Print the data types of each column
    for col in ytrain.columns:
        first_non_null = ytrain[col].dropna().iloc[0] if not ytrain[col].dropna().empty else None
        padding = ' ' * (max(len(c) for c in ytrain.columns) - len(col))
        print(f"  Column '{col}' {padding} -> type: {type(first_non_null).__name__}")

    # Users are in the first column, extract unique users from ytrain
    unique_users = ytrain.iloc[:, 0].unique()
    print("Unique users (labels) in training data:", len(unique_users))

def analyze_class_distribution(ytrain: pd.DataFrame):
    """
    Display how many examples belong to each class in the first column.

    Args:
        ytrain (pd.DataFrame): Training split whose first column contains labels.

    Returns:
        None
    """
    class_counts = ytrain.iloc[:, 0].value_counts()
    class_counts_unique = class_counts.nunique()

    if class_counts_unique == 1:
        print("All classes have the same number of samples.")
        print(f"Each class has {class_counts.iloc[0]} samples.")
    else:
        print("Class distribution in training data:")
        # Group consecutive classes with same count
        prev_count = None
        start_cls = None
        
        for cls, count in class_counts.items():
            if count != prev_count:
                if prev_count is not None:
                    if start_cls == prev_cls:
                        print(f"  Classes {start_cls}: {prev_count} samples")
                    else:
                        print(f"  Classes {start_cls} to {prev_cls}: {prev_count} samples")
                start_cls = cls
                prev_count = count
            prev_cls = cls
        
        # Print the last group
        if start_cls == prev_cls:
            print(f"  Classes {start_cls}: {prev_count} samples")
        else:
            print(f"  Classes {start_cls} to {prev_cls}: {prev_count} samples")

def preserve_letters(text: str, letters: list) -> str:
    """
    Normalize a text string while preserving specific characters (e.g., ``ñ``).

    The function temporarily replaces the target letters with placeholders,
    removes diacritics from the rest of the string, and restores the original
    letters. This is useful when cleaning Spanish tweets for the disinformation
    and polarization analysis without losing language-specific characters.

    Args:
        text (str): Input text to normalize.
        letters (list): Characters to preserve verbatim during normalization.

    Returns:
        str: The normalized text with the specified letters kept intact.
    """
    # marca temporal para no perder la ñ/Ñ
    placeholders = {letter: f"__PLACEHOLDER_{i}__" for i, letter in enumerate(letters)}
    for k, v in placeholders.items():
        text = text.replace(k, v)
    # normalizar y quitar diacríticos
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(ch for ch in text if not unicodedata.combining(ch))
    # restaurar placeholders a ñ/Ñ
    for k, v in placeholders.items():
        text = text.replace(v, k)
    return text

def generate_wordcloud(ytrain: pd.DataFrame):
    """
    Build a word cloud from the tweet column and print frequency diagnostics.

    This helper aggregates the tweet text (assumed to be the last column),
    removes stopwords, saves frequency CSV artifacts for later inspection, and
    renders a matplotlib word cloud to visually inspect common terms in the
    disinformation dataset.

    Args:
        ytrain (pd.DataFrame): Dataset containing tweets in the last column.

    Returns:
        WordCloud: The generated word cloud object.
    """
    # Print most frequent words in the tweets, word cloud and examples by class
    all_text = ' '.join(ytrain.iloc[:, -1].dropna().astype(str).tolist())
    words = all_text.lower().split()
    # Remove punctuation from words
    tokens = preserve_letters(all_text.lower(), ['ñ', 'Ñ'])
    # To remove anything but words (letters, numbers, and underscore)
    words = re.findall(r"(?<!\S)[A-Za-z]\w*", tokens, flags=re.UNICODE)

    clean_words = [w for w in words if len(w) > 2 and w not in stopwords]

    unique_clean_words = set(clean_words)
    print("Número total de palabras únicas (sin stopwords):", len(unique_clean_words))


    df_unique = pd.DataFrame(list(unique_clean_words), columns=["word"])
    df_unique.to_csv("unique_clean_words.csv", index=False, encoding="utf-8")

    word_counts = Counter(clean_words)
    most_common_words = word_counts.most_common(10)
    df_most_common = pd.DataFrame(most_common_words, columns=['word', 'count'])
    # Guardar a CSV
    df_most_common.to_csv("most_common_words.csv", index=False)
    print("Most common words in tweets:")
    for word, count in most_common_words:
        print(f"  '{word}': {count} occurrences")

    word_counts = Counter(clean_words)
    most_common_words = word_counts.most_common(100)

    print("Most common words in tweets (cleaned):")
    for word, count in most_common_words:
        print(f"  '{word}': {count} occurrences")

    clean_text = ' '.join(clean_words)

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        stopwords=stopwords,
        collocations=False,  # avoid duplicated bigrams like "new york"
        normalize_plurals=False
    ).generate(clean_text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Tweets (cleaned)')
    plt.show()
    return wordcloud

if __name__ == "__main__":
    print(f"{'-' * 35}")
    analyze_class_distribution(ytrain)
    print(f"{'-' * 35}")
    # Print text length statistics (from tweets column, which corresponds to the last column)
    text_lengths = ytrain.iloc[:, -1].dropna().apply(len)
    print("Text length statistics:")
    print(f"  Minimum length: {text_lengths.min()}")
    print(f"  Maximum length: {text_lengths.max()}")
    print(f"  Average length: {text_lengths.mean():.2f}")
    print(f"  Median length: {text_lengths.median()}")
    print(f"{'-' * 35}")
    wordcloud = generate_wordcloud(ytrain)
    wordcloud.to_file("wordcloud.png")
