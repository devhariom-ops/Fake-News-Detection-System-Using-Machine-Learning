# Fake News Detection System

An AI-powered natural language processing (NLP) and machine learning (ML) system that classifies news articles as **Real News** or **Fake News** with confidence ratings and credibility metrics. The project includes a model training pipeline, a CLI verification tool, and a premium web interface with a persistent prediction history dashboard using Flask and SQLite.

## Table of Contents
1. [Introduction](#introduction)
2. [Problem Statement](#problem-statement)
3. [Project Structure](#project-structure)
4. [Dataset Description](#dataset-description)
5. [Algorithms Used](#algorithms-used)
6. [Installation & Setup](#installation--setup)
7. [How to Run](#how-to-run)
8. [System Features](#system-features)
9. [Experimental Results](#experimental-results)

---

## Introduction
The rapid spread of misinformation, clickbait, and fake news on social media and news portals poses a significant threat to public trust and decision-making. The **Fake News Detection System** leverages natural language processing (NLP) and machine learning classifiers to assess the credibility of articles instantly, giving users a mathematical rating of news authenticity.

## Problem Statement
Given a news article (its headline or body text), the objective is to build a binary classification system that labels the text as:
*   **Fake News (0)**: Fabrication, clickbait, or deceptive articles.
*   **Real News (1)**: Factual reporting from reputable agencies.

---

## Project Structure
```
FakeNewsDetection/
│
├── dataset/
│   ├── Fake.csv                 # Fake news CSV (Kaggle or generated mock data)
│   └── True.csv                 # Real news CSV (Kaggle or generated mock data)
│
├── models/
│   ├── fake_news_model.pkl      # Best-performing trained ML model
│   └── tfidf_vectorizer.pkl     # Fitted TF-IDF Vectorizer
│
├── static/
│   ├── css/
│   │   └── style.css            # Modern glassmorphism style sheets
│   ├── js/
│   │   └── app.js               # AJAX and gauge pointer interactions
│   └── plots/
│       ├── correlation.png      # Classifier accuracies comparison chart
│       ├── confusion_matrix.png # Best model confusion matrix
│       ├── wordcloud_fake.png   # Word cloud of fake articles
│       ├── wordcloud_true.png   # Word cloud of real articles
│       └── top_words.png        # Bar chart of top 20 frequent terms
│
├── templates/
│   ├── index.html               # Prediction homepage
│   └── dashboard.html           # Admin stats and prediction history
│
├── notebooks/
│   └── analysis.ipynb           # Step-by-step EDA and modeling notebook
│
├── app.py                       # Flask application and SQLite database logic
├── train.py                     # Training and model evaluation script
├── predict.py                   # Interactive CLI prediction tool
├── generate_mock_data.py        # Generates synthetic data for quick setup
├── requirements.txt             # Project requirements list
├── project_report.md            # Academic-style report
└── README.md                    # This document
```

---

## Dataset Description
This system supports the **Kaggle Fake and Real News Dataset** consisting of:
*   `Fake.csv`: ~23,500 records of unverified/sensational news.
*   `True.csv`: ~21,400 records of fact-checked news (primarily from Reuters).
*   Columns expected: `title` (headline), `text` (body content), `subject` (category), and `date` (publication date).

*Note: For immediate out-of-the-box operation, the project generates a high-quality synthetic dataset in the `dataset/` directory if the Kaggle CSV files are not present.*

---

## Algorithms Used
We train and evaluate four machine learning classification algorithms:
1.  **Logistic Regression (`LogisticRegression`)**: Linear model that calculates the log-odds of a class, optimized using L2 regularization. Highly effective for high-dimensional sparse text vectors.
2.  **Naive Bayes (`MultinomialNB`)**: Probabilistic classifier based on Bayes' Theorem, representing word distributions. Standard baseline for document classification.
3.  **Random Forest (`RandomForestClassifier`)**: Ensemble learning model fitting multiple decision tree classifiers to prevent overfitting.
4.  **Passive Aggressive Classifier (`PassiveAggressiveClassifier`)**: Online learning algorithm suitable for large-scale text streams. Adjusts weights dynamically based on classification errors.

---

## Installation & Setup

1.  **Clone or Open the directory**:
    Ensure you are in the project folder containing `requirements.txt`.

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  *(Optional)* **Download Full Kaggle Dataset**:
    If you wish to run on the full Kaggle dataset:
    *   Download from [Kaggle Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset).
    *   Place `Fake.csv` and `True.csv` directly inside the `dataset/` directory.
    *   If not provided, the pipeline generates a starting set of 500 mock news articles automatically.

---

## How to Run

### 1. Train the Models
Build the TF-IDF vocabulary, fit all 4 classifiers, save comparison metrics, and serialize the best model weights:
```bash
python train.py
```
This command outputs accuracy comparison charts, confusion matrices, and word clouds inside the `static/plots/` folder.

### 2. Run Interactive CLI Predictions
Test single predictions on any news text directly from your terminal:
```bash
python predict.py
```
Type any news article body or headline when prompted.

### 3. Launch Flask Web Application
Start the interactive UI:
```bash
python app.py
```
Open your web browser and navigate to `http://127.0.0.1:5000/`.

---

## System Features

### Premium Glassmorphic Web App (`/`)
*   **Live Prediction Grid**: paste articles and click Analyze. Results load instantly via AJAX without page refreshes.
*   **Probability Score Indicator**: shows the numerical percentage confidence of the classification.
*   **Dynamic Credibility Meter**: a color-coded indicator categorizing articles as **Low**, **Medium**, or **High Credibility** based on predictions.

### Admin Dashboard (`/dashboard`)
*   **SQLite-backed Prediction Logs**: persistent logging storing past predictions and timestamps.
*   **System Counters**: view total analysis counts, fake news count, and real news count.
*   **Data Visualizations**: renders word clouds, confusion matrices, and accuracy bar charts generated directly from the model training process.
*   **System Controls**: delete database logs easily via the "Clear System History" button.

---

## Experimental Results
Model performance comparison (evaluated using 80/20 train/test split on synthetic dataset):

| Classifier Model | Accuracy Score | Precision (Real) | Recall (Real) |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | **100.0%** (Best) | 1.00 | 1.00 |
| **Naive Bayes** | **100.0%** | 1.00 | 1.00 |
| **Random Forest** | **100.0%** | 1.00 | 1.00 |
| **Passive Aggressive Classifier** | **100.0%** | 1.00 | 1.00 |

*Note: The mock dataset contains highly distinct lexicons causing classifiers to achieve 100% accuracy quickly. On the full Kaggle dataset, typical accuracies range between 95% - 99% depending on cleaning configurations.*
