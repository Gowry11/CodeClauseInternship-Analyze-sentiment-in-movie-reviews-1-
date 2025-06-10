# CodeClauseInternship-Analyze-sentiment-in-movie-reviews-1-
A beginner-friendly NLP project for classifying movie reviews as positive or negative using NLTK and machine learning. Includes text preprocessing, TF-IDF vectorization, logistic regression classification, and optional K-Means clustering. Built as part of the CodeClause Data Science Internship.


##  Project Overview

- **Goal**: Analyze the sentiment (positive/negative) of movie reviews.
- **Dataset**: NLTK's built-in `movie_reviews` dataset.
- **Tech Stack**: Python, NLTK, Scikit-learn (sklearn), Matplotlib, Seaborn.

##  Project Structure

### 1. Import Libraries & Load Dataset
We use Python libraries like `nltk`, `sklearn`, and `matplotlib`. We load a collection of movie reviews provided by NLTK.

### 2. Preprocess the Text
- Convert text to lowercase.
- Remove punctuation.
- Remove common stopwords (like "the", "is", etc.).
- Lemmatize words (convert to base form, e.g., "running" → "run").

This helps the model understand the meaning better.

### 3. Convert Text to Numbers (TF-IDF)
We use **TF-IDF Vectorization** to convert text into numbers. This helps machine learning algorithms work with text data.

### 4. Train a Sentiment Classifier
We train a **Logistic Regression** model:
- Split the data into training and testing sets.
- Train the model on training data.
- Test it on unseen data to measure accuracy.

### 5. Evaluate the Model
We use accuracy and classification metrics to check how well the model predicts the sentiment.

### 6. Bonus: K-Means Clustering
We apply **K-Means** to try grouping reviews into 2 clusters (positive/negative) without labels. Then we compare those clusters to the real labels.

### 7. Visualization
A simple plot shows how many reviews are positive vs. negative.

---
