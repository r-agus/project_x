Future Work
===========

Future enhancements could further strengthen model performance, topic resolution, and overall system robustness. Below are several potential development lines:

- Apply systematic techniques to estimate the optimal number of clusters (e.g., elbow, silhouette, DBI).  
- Explore deeper or alternative neural architectures, particularly for sequence-based models.  
- Increase training epochs and refine hyperparameters for Transformer-based models.  
- Evaluate multilingual or domain-adapted language models to improve generalization.  
- Test alternative clustering algorithms (e.g., HDBSCAN, spectral clustering, GMM).  
- Improve reproducibility through more structured experiment tracking and versioning.  

Limitations
===========

The project faced several practical and methodological constraints that shaped the final outcomes, some of which were mitigated by reducing dataset size and leveraging cloud or virtual environments such as Kaggle:

- Limited GPU availability restricted model depth, training duration, and hyperparameter search.  
- Memory and RAM constraints affected the processing of large embedding matrices and slowed experimentation.  
- Strong topic imbalance, especially evident in BERTopic outputs.  
- High sensitivity to preprocessing decisions (tokenization, stopword filtering, normalization).  
- Restricted fine-tuning of Transformer models due to computational resource limits.  
- Potential overfitting for rare topics or very small clusters.  
- Limited capacity to run cross-validation or extensive ablation studies.  
- Dependence on third-party environments introduced variability in execution times and resource stability.  

Conclusions
===========

The results offer a structured view of how different modeling strategies capture thematic patterns in political discourse. In summary:

- BERTopic provides more coherent, fine-grained, and interpretable topics than K-Means.  
- K-Means generates broader, more generic clusters that are useful for coarse segmentation.  
- Both approaches confirm that the dataset is heavily dominated by Spanish political content.  
- BERTopic’s outlier detection contributes an interpretability advantage absent in standard clustering.  
- Overall, BERTopic delivers richer topic organization, while K-Means remains a solid and scalable baseline.  
