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
from init import get_data_splits
import matplotlib.pyplot as plt
import seaborn as sns


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

def analyze_class_distribution(ytrain: pd.DataFrame, generate_plots: bool = False):
    """
    Display how many examples belong to each class in the training data (excluding user and tweet text columns).
    
    The training dataframe contains these headers:

    +-----------------------+----------------------------------+
    | Column                | Description                      |
    +=======================+==================================+
    | label                 | user identifier/label            |
    +-----------------------+----------------------------------+
    | gender                | gender classification            |
    +-----------------------+----------------------------------+
    | profession            | professional category            |
    +-----------------------+----------------------------------+
    | ideology_binary       | binary ideology classification   |
    +-----------------------+----------------------------------+
    | ideology_multiclass   | multi-class ideology             |
    |                       | classification                   |
    +-----------------------+----------------------------------+
    | tweet                 | tweet text content               |
    +-----------------------+----------------------------------+

    The distribution of the training dataset is shown in the plot generated when
    ``generate_plots`` is set to ``True``.

    .. image:: _static/class_distribution.png
        :alt: Class distribution plot
        :width: 600px

    Args:
        ytrain (pd.DataFrame): Training split whose columns contain labels.
        generate_plots (bool): Whether to generate a bar plot for class distribution, and save it as PNG.

    Returns:
        None
    """
    # Ensure ytrain is a DataFrame
    if isinstance(ytrain, pd.Series):
        ytrain = ytrain.to_frame()

    label_columns = ytrain.columns[:]

    for col in label_columns:
        class_counts = ytrain[col].value_counts().sort_index()
        print(f"Class distribution for '{col}':")
        for class_label, count in class_counts.items():
            print(f"  Class '{class_label}': {count} examples")

    if generate_plots:
        # Generate a single bar plot for all label columns.
        # Create separate groups for each label column with spacing
        sns.set_theme(style="whitegrid")

        df = (
            ytrain[label_columns]
            .melt(var_name="group", value_name="label")
            .dropna()
        )

        counts = (
            df.groupby(["group", "label"], sort=False)
            .size()
            .reset_index(name="count")
            .reset_index(drop=True)
        )

        gap = 2.0
        bar_w = 0.85
        palette = dict(zip(label_columns, sns.color_palette("tab10", n_colors=len(label_columns))))

        xticks = []
        xticklabels = []
        group_centers = {}

        start = 0.0
        fig, ax = plt.subplots(figsize=(14, 6))

        for group in label_columns:
            sub = counts[counts["group"] == group].reset_index(drop=True)
            n = len(sub)
            if n == 0:
                continue

            xs = start + np.arange(n)
            ax.bar(xs, sub["count"].to_numpy(), width=bar_w, color=palette[group])

            # per-bar tick labels
            xticks.extend(xs.tolist())
            xticklabels.extend(sub["label"].astype(str).tolist())

            # count labels above bars
            for x, c in zip(xs, sub["count"].to_numpy()):
                ax.annotate(
                    f"{int(c):,}",
                    (x, c),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

            group_centers[group] = float(xs.mean())

            if group != label_columns[-1]:
                ax.axvline(xs[-1] + 0.5 + gap / 2, color="0.85", lw=1)

            start = xs[-1] + 1 + gap

        # Axis formatting (bar labels)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation=25, ha="right")
        ax.set_xlabel("")
        ax.set_ylabel("Number of Examples")
        ax.set_title("Class Distribution Across Label Columns")

        # Second bottom axis for group names (avoids overlap)
        ax_group = ax.twiny()
        ax_group.set_xlim(ax.get_xlim())

        ax_group.xaxis.set_ticks_position("bottom")
        ax_group.xaxis.set_label_position("bottom")

        # push group labels below the per-bar labels
        ax_group.spines["bottom"].set_position(("outward", 55))
        ax_group.spines["top"].set_visible(False)
        ax_group.spines["bottom"].set_visible(False)  # hide extra line

        centers = [group_centers[g] for g in label_columns if g in group_centers]
        labels = [g for g in label_columns if g in group_centers]
        ax_group.set_xticks(centers)
        ax_group.set_xticklabels(labels, fontweight="bold")
        ax_group.tick_params(axis="x", length=0, pad=2)

        plt.subplots_adjust(bottom=0.22)

        plt.tight_layout()
        plt.savefig("class_distribution.png", dpi=200, bbox_inches="tight")
        plt.show()

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

def generate_wordcloud(xtrain: pd.DataFrame):
    """
    Build a word cloud from the tweet column and print frequency diagnostics.

    This helper aggregates the tweet text (assumed to be the last column),
    removes stopwords, saves frequency CSV artifacts for later inspection, and
    renders a matplotlib word cloud to visually inspect common terms in the
    disinformation dataset.

    .. image:: _static/wordcloud.png
        :alt: Class distribution plot
        :width: 600px

    Args:
        xtrain (pd.DataFrame): Single-column dataframe with tweet texts.

    Returns:
        WordCloud: The generated word cloud object.
    """
    # Print most frequent words in the tweets, word cloud and examples by class
    all_text = ' '.join(xtrain.dropna().astype(str).tolist())
    words = all_text.lower().split()
    # Remove punctuation from words
    tokens = preserve_letters(all_text.lower(), ['ñ', 'Ñ'])
    # To remove anything but words (letters, numbers, and underscore)
    words = re.findall(r"(?<!\S)[A-Za-z]\w*", tokens, flags=re.UNICODE)

    clean_words = [w for w in words if len(w) > 2 and w not in stopwords]

    unique_clean_words = set(clean_words)
    print("Total number of unique words (without stopwords):", len(unique_clean_words))


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
    xtrain, ytrain, _, yval, _, ytest = get_data_splits()
    print(f"{'-' * 35}")
    # concatenate y matrices to analyze all labels together
    print_data_info(ytrain)
    print(f"{'-' * 35}")
    analyze_class_distribution(pd.concat([ytrain, yval, ytest], ignore_index=True), generate_plots=True)
    print(f"{'-' * 35}")
    # Print text length statistics (from tweets column, which corresponds to the last column)
    text_lengths = xtrain.dropna().apply(len)
    print("Text length statistics:")
    print(f"  Minimum length: {text_lengths.min()}")
    print(f"  Maximum length: {text_lengths.max()}")
    print(f"  Average length: {text_lengths.mean():.2f}")
    print(f"  Median length: {text_lengths.median()}")
    print(f"{'-' * 35}")
    wordcloud = generate_wordcloud(xtrain)
    wordcloud.to_file("wordcloud.png")
