# CodeClauseInternship-Analyze-Sentiment-in-Movie-Reviews

A beginner-friendly NLP + Machine Learning project for analyzing sentiment in movie reviews. This project uses text preprocessing, TF-IDF vectorisation, and a Logistic Regression model to classify movie reviews as either positive or negative. Built as part of the CodeClause Data Science Internship.

## What This Project Does

- Classifies movie reviews as **positive** or **negative**.
- Uses Natural Language Processing (NLP) techniques to clean and process text.
- Converts raw review text into numerical features using **TF-IDF**.
- Trains a **Logistic Regression** classifier on the data.
- Evaluates model performance using accuracy and a classification report.
- Bonus: Applies **K-Means Clustering** to group reviews into clusters.
- Includes visualization of sentiment distribution using Seaborn.

## Technologies Used

- Programming Language: Python  
- Libraries & Tools:
  - [NLTK](https://www.nltk.org/) – for text preprocessing (tokenization, stopwords, lemmatization)
  - [Scikit-learn](https://scikit-learn.org/) – for vectorization, classification, evaluation, and clustering
  - [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/) – for visualizing results

## Project Structure

### 1. Import Libraries & Load Dataset
The dataset used is the **movie_reviews** corpus available via the NLTK library. Each review is labeled as either `pos` (positive) or `neg` (negative).

### 2. Preprocess the Text
- Convert to lowercase
- Remove punctuation
- Tokenize using NLTK
- Remove stopwords
- Lemmatize using WordNetLemmatizer

### 3. Vectorize the Text (TF-IDF)
Text data is transformed into numerical feature vectors using **TF-IDF** with a maximum of 5000 features.

### 4. Train Classifier
- **Logistic Regression** is used to classify the reviews.
- The dataset is split into training and testing sets.
- Accuracy and classification report are printed after evaluation.

### 5. Apply K-Means Clustering (Bonus)
- Unsupervised K-Means clustering is applied to the same data.
- A confusion matrix is shown to compare actual vs. clustered labels.

### 6. Visualize Sentiment Distribution
- Uses Seaborn to plot the distribution of positive and negative reviews.

## Sample Results

- **Accuracy** (Logistic Regression): ~85–90%
- **Classification Report** shows precision, recall, F1-score
- **K-Means Confusion Matrix** gives a rough clustering overview

## How to Run This Project

**Clone the Repository**:
```bash
git clone https://github.com/Gowry11/CodeClauseInternship-Analyze-Sentiment-in-Movie-Reviews-1-.git
```
**Run the Notebook**:
- Open Sentiment_Analysis_CodeClause.ipynb in Jupyter Notebook
- Run all cells step-by-step from top to bottom

 ## Concepts Covered
- Basics of NLP and text preprocessing
- Using TF-IDF to convert text into features
- Building a simple yet effective ML model
- Understanding classification performance using precision, recall, F1-score
- Applying basic clustering to explore unsupervised learning

## Developed By  
Gowry P P

