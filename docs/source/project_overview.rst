Final Project Overview
======================

Analysis of ideological polarization on social networks based on the spread of disinformation content.

General Description
-------------------
The expansion of disinformation on social networks has transformed the way individuals access information and shape their opinions. Numerous studies indicate that the spread of false or manipulated content not only affects the perception of reality, but can also amplify ideological polarization by reinforcing preexisting beliefs within homogeneous communities.

This project examines the relationship between the spread of disinformation and the structure of polarized communities on social networks (e.g., Twitter/X or Facebook). The central hypothesis is that specific patterns of disinformation function as catalysts of polarization, deepening the divide between social and political groups.

Teams of three or four members may select the most suitable database and analytical approach, within the methodological boundaries and components defined below. The project has a maximum score of **3 points**, broken down as **2.25 points** for the basic project and **0.75 points** for the extension. The submission deadline is **December 10, 2025, at 23:59**.

Specific Objectives
-------------------
- Identify posts with potentially disinformative content using machine learning techniques and language models.
- Analyze the relationship between the diffusion of disinformation and users' ideological polarization.
- Explore sentiment analysis and stance detection models as complementary tools for ideological characterization.

Modeling and Evaluation
-----------------------
Train and evaluate predictive classification or regression models using at least two of the following strategies:

- A neural network implemented in PyTorch.
- At least one other Scikit-learn algorithm (for example, K-NN, SVM, Random Forest, Logistic Regression, etc.).
- A pre-trained Transformer model fine-tuned using the Hugging Face Transformers library.

In all cases, adequate validation of the models must be performed, including explicit separation into training, validation, and test sets.

Comparative Evaluation
----------------------
Teams are expected to:

- Evaluate the models using appropriate metrics (accuracy, F1, ROC-AUC, etc.).
- Analyze performance differences across vector representations and model architectures.
- Interpret the results in relation to the initial hypothesis regarding disinformation and polarization.

Extension Project
-----------------
The extension work is fully open-ended, and you should expand on the basic project in any appropriate direction. Examples include:

- **Thematical analysis of disinformation:** Apply topic modeling or clustering methods on contextualized embeddings to detect recurring disinformation themes.
- **Generative modeling for disinformation analysis:** Explore the ability of LLMs to generate synthetic posts that mimic disinformative or polarized content. Compare outputs from fine-tuned models and prompting-based generation (e.g., LLama). Analyze whether generated text exhibits patterns observed in real data.
- **Lexicon & URL-domain baseline for disinformation cues:** Build a simple rule-based baseline using keyword/hashtag lists and linked URL domains (e.g., known fact-checkers vs. low-credibility sites), and compare its performance against the models created in the basic project.
- **Correlational analysis of polarization:** Investigate how exposure to disinformation or rumour-related content relates to stronger emotional expression, more negative or disagreeing language, and other linguistic signs of polarization in social media discussions.

Take this list as a suggestion; any other topic is acceptable as long as it fits within the course's scope. Originality in the choice will be valued. Avoid overly ambitious extensions that could compromise timely delivery. If in doubt about suitability, consult the instructor. The extension work constitutes **0.75 points** of the final grade.

Project Submission
------------------
The project submission will use GitHub for collaboration and version control and must include:

- Fully functional and documented code.
- A README.md file (or a GitHub Pages site) serving as the project report, including:

  - A description of the problem and the dataset.
  - An explanation of the methodologies used.
  - Experimental results and discussion.
  - Conclusions.

- All files or scripts necessary to reproduce the results.

One student from each group must upload the link to the shared repository to Aula Global. The repository must be public to allow the instructors to review and reproduce the work. Documentation pages may not be edited after the submission deadline; evaluation will be based on the last commit made before the due date.

Authorship
----------
Documentation must adhere to proper authorship acknowledgment. If external code snippets or other materials are used, they must be clearly specified in the report.

Evaluation
----------
The project will be evaluated based on the submitted documentation and an individual presentation during the week of **16 December**. Evaluation criteria are summarized below:

Basic Project (2.25 points)
    - Methodology (0.75)
    - Quality of documentation (0.3)
    - Code quality (0.3)
    - Presentation quality (0.5)
    - Responses to comments on the presentation (0.4)

Extension (0.75 points)
    - Originality (0.25)
    - Quality of the work (0.5)
