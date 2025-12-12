# thematic_analysis.py
import pandas as pd
import numpy as np
import json
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from stopwords import stopwords
from bertopic import BERTopic
from umap import UMAP
import hdbscan


def cluster_with_kmeans(texts, embeddings, n_clusters=5):
    """
    Aplica PCA + KMeans para agrupar textos en temas.
    """
    pca = PCA(n_components=50, random_state=42)
    embeddings_reduced = pca.fit_transform(embeddings)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings_reduced)
    
    df_topics = pd.DataFrame({"tweet": texts, "cluster": clusters})
    
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
    BERTopic ajustado para producir MÁS clusters, evitando un Topic 0 gigante.
    """

    # === 1. UMAP (reduce dimensión pero separa por vecindarios pequeños) ===
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric='cosine',
        random_state=42
    )

    # === 2. HDBSCAN (clave para aumentar número de clusters) ===
    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=10,            # clusters pequeños
        min_samples=5,                  # más sensibilidad
        metric='euclidean',
        cluster_selection_method='eom',
        cluster_selection_epsilon=0.05, # divide clusters grandes
        prediction_data=True
    )

    # === 3. Vectorizer ===
    vectorizer_model = CountVectorizer(stop_words=list(stopwords))

    # === 4. BERTopic ===
    topic_model = BERTopic(
        language=language,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,  # sustituye al clustering por defecto
        vectorizer_model=vectorizer_model,
        min_topic_size=10,            # permite temas pequeños
        nr_topics=None,               # NO fusionar automáticamente temas
        verbose=True
    )

    topics, probs = topic_model.fit_transform(texts, embeddings)

    return topic_model, topics, probs



def save_clusters_to_json(df_topics, filename="clusters_kmeans.json", max_tweets=50):
    output_data = {}
    clusters = sorted(df_topics['cluster'].unique())
    
    for cluster_id in clusters:
        tweets = df_topics[df_topics['cluster'] == cluster_id]['tweet'].tolist()
        if max_tweets:
            tweets = tweets[:max_tweets]
        output_data[int(cluster_id)] = tweets
        
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    
    print(f"Archivo de inspección guardado: {filename}")



# =======================================================
#                  EJEMPLO DE USO COMPLETO
# =======================================================
if __name__ == "__main__":
    import TextVectorRepresentation as TV
    
    path = "Datasets/EvaluationData/politicES_phase_2_train_public.csv"
    data = TV.load_data(path)
    data = data.sample(n=10000, random_state=42).reset_index(drop=True)
    
    train_data, val_data, test_data = TV.divide_train_val_test(data)
    X_train, y_train = TV.separate_x_y_vectors(train_data)
    X_val, y_val = TV.separate_x_y_vectors(val_data)
    X_test, y_test = TV.separate_x_y_vectors(test_data)
    
    texts = pd.concat([X_train, X_val, X_test]).reset_index(drop=True)

    x_BERT_train, x_BERT_val, x_BERT_test = TV.vectorRepresentation_BERT(X_train, X_val, X_test)
    embeddings = np.vstack([x_BERT_train, x_BERT_val, x_BERT_test])

    # -------------------------------
    # A) KMeans
    # -------------------------------
    print("--- Ejecutando KMeans ---")
    df_topics, top_words = cluster_with_kmeans(texts, embeddings, n_clusters=10)
    
    print("Top words per cluster (KMeans):")
    for cluster, words in top_words.items():
        print(f"Cluster {cluster}: {', '.join(words)}")

    cluster_counts = df_topics['cluster'].value_counts().sort_index()
    print("\nNúmero de tweets por cluster (KMeans):")
    for cluster_id, count in cluster_counts.items():
        print(f"Cluster {cluster_id}: {count} tweets")

    save_clusters_to_json(df_topics, filename="clusters_kmeans_inspection.json", max_tweets=50)

    # -------------------------------
    # B) BERTopic con más clusters
    # -------------------------------
    print("\n--- Ejecutando BERTopic (más clusters) ---")
    topic_model, topics, probs = cluster_with_bertopic(texts.tolist(), embeddings)
    
    print("\nBERTopic info:")
    topic_info = topic_model.get_topic_info()
    print(topic_info)

    bertopic_counts = pd.Series(topics).value_counts().sort_index()
    print("\nNúmero de tweets por topic (BERTopic):")
    for topic_id, count in bertopic_counts.items():
        print(f"Topic {topic_id}: {count} tweets")
        
    topic_info.to_json("bertopic_info_representativos.json", orient='records', indent=4, force_ascii=False)
    print("Archivo BERTopic guardado: bertopic_info_representativos.json")
