Exploratory Analysis of the Dataset
===================================

This section explores the dataset used in this project, which is the `POLITiCES 2023 dataset <https://codalab.lisn.upsaclay.fr/competitions/10173#learn_the_details-get_starting_kit>`_, focusing on understanding the distribution of ideological polarization among users and the prevalence of disinformation content.

Distribution of Ideological Polarization
----------------------------------------
The training dataset contains 180.000 rows, containing the following columns:

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

The class distribution is shown in the histogram below, where it can be seen that `gender` and `profession` are less balanced compared to `ideology_binary` and `ideology_multiclass`.

.. image:: _static/class_distribution.png
   :alt: Histogram of Ideological Polarization Scores
   :width: 600px
   :align: center

A preliminary text analysis is performed, including a word cloud visualization of the most frequent words in the tweets, which can be seen below.

.. image:: _static/wordcloud.png
   :alt: Word Cloud of Most Frequent Words
   :width: 600px
   :align: center