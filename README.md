![Status](https://img.shields.io/badge/Status-Completed-success)
![Type](https://img.shields.io/badge/Project-NLP%20%7C%20Sentiment%20Analysis-blueviolet)
![Domain](https://img.shields.io/badge/Domain-Social%20Media-informational)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-Baseline%20Established-brightgreen)
![Data Source](https://img.shields.io/badge/Data-Kaggle-00AEEF?logo=kaggle&logoColor=white)
![Classes](https://img.shields.io/badge/Classes-4%20Labels-yellow)

---
PRODIGY_DS__-04
**Author:** Adeyeye Blessing Temidayo  
**CIN:** PIT/DEC25/10676  

## 🐦 Twitter Sentiment Analysis and Classification
Author: Adeyeye Blessing Temidayo

## 📌 Project Overview
This project performs sentiment classification on social media data, transitioning from raw, unstructured text to a predictive model using Natural Language Processing (NLP) techniques. The goal is to build a machine learning pipeline capable of categorizing tweets into four distinct labels: Positive, Negative, Neutral, and Irrelevant.

## 🎯 Key Objectives
Sentiment Classification: Analyze Twitter data to support policy analysis, public health communication, and brand perception tracking.

End-to-End Pipeline: Demonstrate the standard Data Science Life Cycle, including data preprocessing, feature engineering, and model evaluation.

Predictive Modeling: Leverage TF-IDF vectorization and Logistic Regression to classify text sentiment accurately.

## 🛠️ Technical Implementation
## Data Pipeline
- Preprocessing: Handled missing values (removed ~0.009% null tweet content) and performed text cleaning using regular expressions.

- NLP Techniques: * Tokenization & Stemming: Used NLTK's PorterStemmer to reduce words to their root forms.

- Stopword Removal: Filtered out common English stopwords to focus on meaningful sentiment-bearing words.

- Vectorization: Implemented TF-IDF (Term Frequency-Inverse Document Frequency) to transform raw text into a weighted numerical matrix.

#### Machine Learning Models
Logistic Regression (Primary Model): Selected for its efficiency and strong baseline performance in text classification.

Optimization: Utilized GridSearchCV and a Scikit-Learn Pipeline to automate the workflow and tune hyperparameters.

## 🚀 How to Use
1. **Clone the repo:** `git clone https://github.com/Temidayo23/PRODIGY_DS__-04`
2. **Install requirements:** `pip install -r requirements.txt`
3. **Run the Notebook:** Open `Twitter_analysis.ipynb` in Jupyter or Google Colab.

----

## 📊 Evaluation & Results
- Confusion Matrix: Evaluated the model's ability to distinguish between all four sentiment categories.

- Classification Report: Measured performance using precision, recall, and F1-score across all labels.

- Efficiency: The current Logistic Regression model provides a robust baseline for sentiment analysis on a dataset of ~74,682 records.
  
- Hyperparameter tuning further enhances model generalization
  
  (Refer to notebook outputs for detailed metrics and visualizations.)
----
## 📂 Repository Structure
```text
├── Twitter_analysis.ipynb   # Main analysis and modeling notebook
├── twitter_training.csv      # Dataset (sourced from Kaggle)
└── README.md                 # Project documentation
```
----
## 🧪 Methodology

### 1. Data Preprocessing
- Removal of URLs, mentions, hashtags, punctuation, and special characters
- Text normalization (lowercasing)
- Tokenization and stopword removal
- Stemming using Porter Stemmer

### 2. Exploratory Data Analysis (EDA)
- Class distribution visualization
- Frequent word analysis
- Basic statistical summaries

### 3. Feature Engineering
- TF-IDF (Term Frequency–Inverse Document Frequency) vectorization
- N-gram support for improved contextual understanding

### 4. Model Building
- Logistic Regression classifier
- Scikit-learn Pipeline for reproducibility
- Train-test data splitting

### 5. Model Evaluation
- Accuracy score
- Confusion matrix visualization
- Classification report (precision, recall, F1-score)

### 6. Model Optimization
- Hyperparameter tuning using GridSearchCV

## 📈 Future Scope
Advanced Modeling: Implementing Transformer-based models like BERT to capture deeper semantic relationships.

Resampling: Using techniques like SMOTE to address data imbalance in "Irrelevant" and "Neutral" samples.

Real-time Integration: Deploying via Streamlit or Flask to monitor live Twitter feeds.

## 📧 Contact
Adeyeye Blessing Temidayo (adeyeyeblessing2017@gmail.com) Feel free to reach out for collaborations or questions regarding this analysis!
## 📄 License
MIT License Copyright (c) 2026 Adeyeye Blessing Temidayo

Permission is hereby granted, free of charge... (See full text in repository)

Disclaimer: This project was developed as part of a Data Science project with Prodigy InfoTech and uses the Kaggle Twitter Entity Sentiment Analysis Dataset.
