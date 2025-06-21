# CodeClauseInternship-Analyze-sentiment-in-movie-reviews

A beginner-friendly NLP project for classifying movie reviews as positive or negative using NLTK and machine learning. Includes text preprocessing, TF-IDF vectorisation, logistic regression classification, and optional K-Means clustering. Built as part of the **CodeClause Data Science Internship**.

---

## What This Project Does

- Uses Natural Language Processing (NLP) techniques to analyse the sentiment of movie reviews.
- Classifies reviews as either **positive** or **negative** using a Logistic Regression model.
- Applies **TF-IDF Vectorisation** to transform text into numerical features.
- Optionally uses **K-Means Clustering** to group reviews without labels.
- Includes visualisations to understand data distribution.

---

## Technologies Used

- **Programming Language**: Python  
- **Libraries & Tools**:
  - [NLTK](https://www.nltk.org/) – for natural language processing
  - [Scikit-learn](https://scikit-learn.org/) – for machine learning models and vectorisation
  - [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/) – for visualization

---

## Project Structure

### 1. Import Libraries & Load Dataset
We use Python libraries like `nltk`, `sklearn`, and `matplotlib`. We load the dataset `movie_reviews` from NLTK.

### 2. Preprocess the Text
- Convert text to lowercase
- Remove punctuation and stopwords
- Apply lemmatisation to reduce words to their base form

### 3. Convert Text to Numbers (TF-IDF)
Text is converted into numerical form using TF-IDF so that it can be used by machine learning algorithms.

### 4. Train a Sentiment Classifier
- Split the dataset into training and testing sets
- Use **Logistic Regression** for classification

### 5. Evaluate the Model
Check model performance using the accuracy score and the classification report.

### 6. Bonus: K-Means Clustering
Try to group reviews into 2 clusters using **unsupervised learning** and compare with actual labels.

### 7. Visualization
Plot the distribution of positive and negative reviews.

---

## How to Run This Project

**Clone the Repository**:
   ```bash
   git clone https://github.com/Gowry11/CodeClauseInternship-Analyze-sentiment-in-movie-reviews-1-.git

