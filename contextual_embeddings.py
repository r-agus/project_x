# thematic_analysis.py
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from stopwords import stopwords
from bertopic import BERTopic

def cluster_with_kmeans(texts, embeddings, n_clusters=5):
    """
    Aplica PCA + KMeans para agrupar textos en temas.
    
    Args:
        texts: Serie o lista de textos.
        embeddings: numpy array de embeddings (num_texts, dim).
        n_clusters: número de clusters/temas.
    
    Returns:
        df_topics: DataFrame con columnas 'tweet' y 'cluster'.
        top_words_per_cluster: diccionario cluster -> palabras más representativas.
    """
    # Reducir dimensionalidad
    pca = PCA(n_components=50, random_state=42)
    embeddings_reduced = pca.fit_transform(embeddings)
    
    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings_reduced)
    
    df_topics = pd.DataFrame({"tweet": texts, "cluster": clusters})
    
    # TF-IDF para obtener palabras más representativas
    vectorizer = TfidfVectorizer(max_features=1000, stop_words=list(stopwords))
    X_tfidf = vectorizer.fit_transform(df_topics["tweet"])
    
    top_words_per_cluster = {}
    for cluster_id in range(n_clusters):
        idx = (df_topics["cluster"] == cluster_id).to_numpy()
        tfidf_cluster = X_tfidf[idx].mean(axis=0)
        top_words = np.array(vectorizer.get_feature_names_out())[tfidf_cluster.A1.argsort()[::-1][:10]]
        top_words_per_cluster[cluster_id] = top_words.tolist()
    
    return df_topics, top_words_per_cluster

def cluster_with_bertopic(texts, embeddings, language="multilingual"):
    """
    Aplica BERTopic para detectar temas automáticamente.
    
    Args:
        texts: Lista de textos.
        embeddings: numpy array de embeddings.
        language: idioma de los textos (para BERTopic).
    
    Returns:
        topic_model: modelo BERTopic entrenado.
        topics: lista de tema asignado a cada texto.
        probs: probabilidades de pertenencia.
    """
    topic_model = BERTopic(language=language)
    topics, probs = topic_model.fit_transform(texts, embeddings)
    return topic_model, topics, probs

# =========================
# EJEMPLO DE USO
# =========================
if __name__ == "__main__":
    import TextVectorRepresentation as TV
    path = "Datasets/EvaluationData/politicES_phase_2_train_public.csv"
    data = TV.load_data(path)
    data = data.head(2000)  # O especifica el número de filas que necesites
    train_data, val_data, test_data = TV.divide_train_val_test(data)
    X_train, y_train = TV.separate_x_y_vectors(train_data)
    X_val, y_val = TV.separate_x_y_vectors(val_data)
    X_test, y_test = TV.separate_x_y_vectors(test_data)
    # Concatenar todos los textos
    texts = pd.concat([X_train, X_val, X_test]).reset_index(drop=True)

    # =========================
    # Usar embeddings BERT
    # =========================
    x_BERT_train, x_BERT_val, x_BERT_test = TV.vectorRepresentation_BERT(X_train, X_val, X_test)
    embeddings = np.vstack([x_BERT_train, x_BERT_val, x_BERT_test])

    # KMeans
    df_topics, top_words = cluster_with_kmeans(texts, embeddings, n_clusters=10)
    print("Top words per cluster (KMeans):")
    for cluster, words in top_words.items():
        print(f"Cluster {cluster}: {', '.join(words)}")

    # BERTopic
    topic_model, topics, probs = cluster_with_bertopic(texts.tolist(), embeddings)
    print("\nBERTopic info:")
    print(topic_model.get_topic_info())
    # =========================
# KMeans: imprimir número de tweets por cluster
# =========================
    cluster_counts = df_topics['cluster'].value_counts().sort_index()
    print("\nNúmero de tweets por cluster (KMeans):")
    for cluster_id, count in cluster_counts.items():
        print(f"Cluster {cluster_id}: {count} tweets")

    # =========================
    # BERTopic: imprimir número de tweets por topic
        # =========================
    import pandas as pd
    bertopic_counts = pd.Series(topics).value_counts().sort_index()
    print("\nNúmero de tweets por topic (BERTopic):")
    for topic_id, count in bertopic_counts.items():
        print(f"Topic {topic_id}: {count} tweets")
