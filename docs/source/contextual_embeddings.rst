Extension for Contextual Embeddings
===================================

As part of the project extension, a new module ``contextual_embeddings.py`` has been developed to implement tweet clustering using contextual embeddings derived from BERT. This module provides functions for clustering with KMeans and BERTopic, as well as functionality to export clustered tweets to JSON files for further inspection.

This extension enables more sophisticated analysis of tweet content by leveraging contextual embeddings, which capture semantic meaning more effectively than traditional methods. The module facilitates the identification of topics and themes within the tweet dataset.

Purpose
-------

The primary objective of this module is to perform **automatic topic discovery** within large collections of tweets. Rather than requiring manual examination of thousands of tweets to identify discussion themes, the system groups semantically similar tweets together and extracts the key topics from each group.

Methodology
-----------

The module implements two distinct approaches for topic discovery:

**KMeans Clustering Approach**

1. **Dimensionality Reduction**: Tweets are first converted into high-dimensional numerical representations (embeddings) that encode their semantic content. These representations are then reduced to a lower-dimensional space using Principal Component Analysis (PCA), preserving the most relevant information while improving computational efficiency.

2. **Cluster Formation**: The KMeans algorithm partitions all tweets into a predefined number of groups (e.g., 5 or 10 topics). Tweets with similar semantic content are assigned to the same cluster.

3. **Topic Characterization**: For each cluster, the system identifies the ten most representative terms using TF-IDF weighting. These terms serve as descriptors for each discovered topic.

**BERTopic Approach**

This method employs a more sophisticated pipeline that:

- Automatically determines the optimal number of topics without requiring manual specification
- Identifies smaller, more granular topics through density-based clustering
- Utilizes advanced dimensionality reduction techniques (UMAP) to better preserve local semantic relationships

This approach requires greater computational resources but typically yields more coherent and interpretable results.

Output
------

The analysis produces the following outputs:

- **Topic Descriptions**: Each topic is characterized by its most representative terms
- **Tweet Assignments**: Each tweet is assigned to a specific topic, enabling examination of topic composition
- **Inspection Files**: JSON files containing sample tweets from each topic for manual review

As an illustrative example, analysis of political tweets might reveal topics such as:

- Topic 1: "economy", "taxes", "budget", "spending" (economic policy discussions)
- Topic 2: "vote", "election", "candidate", "polls" (electoral discourse)
- Topic 3: "healthcare", "hospital", "insurance" (health policy discussions)

Applications
------------

This automatic topic discovery functionality serves several purposes:

- **Public Opinion Analysis**: Rapid identification of dominant themes in public discourse
- **Research Applications**: Pattern detection in large datasets without manual annotation
- **Content Classification**: Automatic categorization of content by thematic similarity
- **Trend Identification**: Detection of frequently discussed topics within a corpus

Experimental Results
--------------------

The module was executed on a sample of 10,000 tweets from the PoliticES dataset (``politicES_phase_2_train_public.csv``). The following results were obtained:

**KMeans Clustering Results**

Using KMeans with 10 clusters, the algorithm partitioned the tweets into groups of varying sizes. The distribution of tweets across clusters is shown below:

.. list-table:: KMeans Cluster Distribution
   :header-rows: 1
   :widths: 20 30 50

   * - Cluster
     - Tweet Count
     - Top Representative Terms
   * - 0
     - 1,043
     - españa, gobierno, años, madrid, millones, euros, ucrania
   * - 1
     - 915
     - gobierno, hoy, españa, ley, congreso, años, madrid
   * - 2
     - 911
     - hoy, gracias, españa, gran, mañana, entrevista, enhorabuena
   * - 3
     - 952
     - si, usted, ver, ser, puede, así, mismo
   * - 4
     - 888
     - gobierno, hoy, madrid, ley, congreso, ahora, partido
   * - 5
     - 1,119
     - gracias, si, bien, así, abrazo, verdad, siempre
   * - 6
     - 1,430
     - hoy, años, madrid, día, ser, gran, dos
   * - 7
     - 1,153
     - si, hoy, ahora, día, gracias, aquí, nunca
   * - 8
     - 974
     - si, puede, ser, gobierno, solo, españa, ley
   * - 9
     - 615
     - gracias, hoy, día, españa, gobierno, vida, trabajo

The KMeans results reveal that the largest cluster (Cluster 6) contains 1,430 tweets, while the smallest (Cluster 9) contains 615 tweets. The representative terms suggest topics related to Spanish politics, government affairs, and social interactions.

**BERTopic Results**

BERTopic automatically identified 16 distinct topics (plus one outlier category). The distribution is notably different from KMeans:

.. list-table:: BERTopic Topic Distribution
   :header-rows: 1
   :widths: 15 20 65

   * - Topic
     - Tweet Count
     - Thematic Description
   * - -1
     - 272
     - Outliers (unclassified tweets)
   * - 0
     - 8,832
     - General political discourse (gobierno, españa, gracias)
   * - 1
     - 224
     - Congressional and legislative matters (gobierno, madrid, congreso, ley)
   * - 2
     - 200
     - Annual reports and national affairs (año, gobierno, españa, país, millones)
   * - 3
     - 50
     - Regional topics - Sevilla, women's issues
   * - 4
     - 48
     - Constitutionalism and corruption
   * - 5
     - 48
     - Political conflicts and institutional matters
   * - 6
     - 46
     - Constitutional and party politics
   * - 7
     - 44
     - Occupation issues and competitiveness
   * - 8
     - 44
     - Judicial reform (CGPJ) and government
   * - 9
     - 37
     - Legislation and digital currencies
   * - 10
     - 36
     - Obituaries and commemorations
   * - 11
     - 33
     - Regional government (Andalucía leadership)
   * - 12
     - 27
     - Rural depopulation issues
   * - 13
     - 24
     - Women's issues and crime
   * - 14
     - 20
     - Financial and political accountability
   * - 15
     - 15
     - Healthcare and residency policies

**Analysis of Results**

Several observations can be drawn from the experimental results:

1. **Topic Granularity**: BERTopic identifies more specific and granular topics compared to KMeans. While KMeans produces relatively balanced clusters, BERTopic reveals a dominant general topic (Topic 0 with 8,832 tweets) alongside multiple smaller, more specialized topics.

2. **Thematic Coherence**: The BERTopic results demonstrate higher thematic coherence. Topics such as "judicial reform (CGPJ)", "rural depopulation", and "healthcare policies" are clearly identifiable, whereas KMeans clusters show more overlap in their representative terms.

3. **Outlier Handling**: BERTopic explicitly identifies 272 outlier tweets (Topic -1) that do not fit well into any topic, providing transparency about classification uncertainty.

4. **Political Content**: Both methods confirm that the dataset consists primarily of Spanish political discourse, with recurring themes including government affairs, legislative processes, regional politics, and social issues.

5. **Distribution Imbalance**: The BERTopic results show a highly imbalanced distribution, with 88.3% of tweets falling into the general topic (Topic 0). This suggests that most tweets share common political discourse characteristics, while specific issues appear in smaller, distinct clusters.
