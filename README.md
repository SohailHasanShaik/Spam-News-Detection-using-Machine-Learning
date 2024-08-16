# Spam News Detection Using Machine Learning

This project implements a spam news detection system using machine learning techniques. It uses Natural Language Processing (NLP) to preprocess text data and employs a Naive Bayes classifier for spam detection.

## Project Overview

This project aims to classify news articles into "True" or "Fake" categories using a dataset of news articles. The process involves:
1. **Data Preprocessing**: Cleaning and preparing text data.
2. **Feature Extraction**: Converting text data into numerical features using TF-IDF.
3. **Model Training**: Training a Naive Bayes classifier on the processed data.
4. **Model Evaluation**: Assessing the performance of the trained model.

## Installation

To run this project, you need Python and the following libraries:

- `pandas`
- `numpy`
- `nltk`
- `scikit-learn`

You can install the required libraries using pip:

```bash
pip install pandas numpy nltk scikit-learn
```

## Usage

### Data Preparation

1. **Load Data**: Load your news datasets into the `True_news` and `Fake_news` DataFrames.
2. **Preprocessing**: Clean the text data using NLP techniques.

### Running the Code

1. **Prepare Your Dataset**: Place your CSV files (`True_news.csv` and `Fake_news.csv`) in the same directory as the script.
2. **Run the Script**: Execute the script in your Python environment.

```bash
python your_script_name.py
```

3. **Interact with the Model**: You can enter a news headline to see if it is classified as true or fake.

## Data

- **True News Dataset**: CSV file containing news articles labeled as true.
- **Fake News Dataset**: CSV file containing news articles labeled as fake.

Ensure the CSV files have a column named `text` containing the news articles.

## Model Training

The model is trained using the Naive Bayes algorithm, which is suitable for text classification tasks. The TF-IDF vectorizer converts text data into numerical features, and the Naive Bayes classifier is trained on these features.

## Model Evaluation

The performance of the model is evaluated using accuracy score. You can also explore other metrics such as precision, recall, and F1-score.
