Text Vector Representation
==========================

This section describes the text vector representations used as input to the downstream classifiers. We implement three complementary strategies — TF-IDF, Word2Vec and contextual BERT embeddings — all built on the same tweet corpus and train/validation/test splits, so that their impact on performance can be compared fairly.

TF-IDF
----------------

For TF-IDF, we use a TfidfVectorizer with up to 5 000 features and unigram/bigram n-grams, removing Spanish stopwords. The resulting representation is a high-dimensional sparse matrix that captures how informative each token is in a tweet relative to the whole corpus, and is particularly well suited for linear models such as Logistic Regression and Linear SVM.

Word2Vec
----------------

The Word2Vec representation is based on dense, low-dimensional embeddings trained on the preprocessed training tweets. We tokenize each tweet (lowercasing, accent/ punctuation handling and stopword removal) and train a skip-gram Word2Vec model. Tweet vectors are then obtained by averaging the embeddings of all in-vocabulary tokens, while tweets with no valid tokens are mapped to zero vectors. This yields compact, continuous representations that encode distributional semantic information.


BERT Embeddings
----------------
Finally, for contextual embeddings we rely on the multilingual Transformer model bert-base-multilingual-cased. Each tweet is tokenized with the corresponding BERT tokenizer, passed through the encoder, and represented by the mean of the last-layer token embeddings. In contrast to TF-IDF and Word2Vec, these contextual embeddings allow the same word to receive different vectors depending on its surrounding context, which is particularly relevant in short, noisy political tweets.