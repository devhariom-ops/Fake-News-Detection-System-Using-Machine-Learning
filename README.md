# 📰 Fake News Detection System Using Machine Learning & NLP

## 📌 Project Overview

The **Fake News Detection System** is an intelligent Machine Learning and Natural Language Processing (NLP) application designed to automatically identify whether a news article is **Fake** or **Real**. With the rapid expansion of digital media and social networking platforms, misinformation spreads faster than ever before, making automated verification systems increasingly important.

This project utilizes a publicly available Kaggle dataset containing thousands of real and fake news articles. The collected data undergoes extensive preprocessing, text cleaning, and feature extraction using **TF-IDF (Term Frequency–Inverse Document Frequency)** techniques. Multiple Machine Learning algorithms are trained and evaluated to determine the most accurate model for news classification.

The final solution is deployed as a user-friendly Flask web application where users can enter news content and instantly receive a prediction along with a confidence score.

---

# 🎯 Problem Statement

The rise of social media platforms has significantly increased the circulation of misleading information and fake news. Such misinformation can influence public opinion, create social unrest, and spread confusion among people.

Traditional fact-checking methods are often manual, time-consuming, and unable to keep pace with the volume of content generated daily. Therefore, there is a need for an automated and scalable system capable of detecting fake news quickly and accurately.

This project addresses this challenge by leveraging Machine Learning and NLP techniques to classify news articles based on their textual content.

---

# 🎯 Project Objectives

* Develop an automated system to classify news articles as Fake or Real.
* Apply Natural Language Processing techniques for text analysis.
* Perform data preprocessing and feature engineering on textual datasets.
* Train and compare multiple Machine Learning algorithms.
* Evaluate model performance using standard classification metrics.
* Develop a responsive Flask-based web application.
* Store prediction history using a database system.
* Provide confidence scores for prediction transparency.

---

# ✨ Key Features

## 👤 User Features

✔ News Article Input Interface

✔ Instant Fake/Real News Prediction

✔ Confidence Score Display

✔ Clean and Responsive User Interface

✔ Real-Time Prediction Results

✔ Prediction History Tracking

✔ User-Friendly Web Experience

---

## 🔐 Admin Features

✔ Total Predictions Dashboard

✔ Fake News Detection Statistics

✔ Real News Detection Statistics

✔ SQLite Database Management

✔ Historical Prediction Records

✔ System Monitoring Dashboard

---

# 🛠 Technology Stack

## Programming Language

* Python

## Machine Learning Libraries

* Scikit-Learn
* Joblib

## Natural Language Processing

* NLTK
* Porter Stemmer

## Data Analysis & Processing

* Pandas
* NumPy

## Data Visualization

* Matplotlib
* Seaborn
* WordCloud

## Web Development

* Flask
* HTML5
* CSS3
* Bootstrap 5

## Database

* SQLite

---

# 📂 Project Architecture

```text
FakeNewsDetection/
│
├── dataset/
│   ├── Fake.csv
│   └── True.csv
│
├── models/
│   ├── fake_news_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── static/
│   ├── css/
│   ├── js/
│   └── images/
│
├── templates/
│   ├── index.html
│   ├── history.html
│   └── dashboard.html
│
├── database/
│   └── predictions.db
│
├── train.py
├── predict.py
├── app.py
├── requirements.txt
├── README.md
└── notebook.ipynb
```

---

# 📊 Dataset Information

## Dataset Source

Fake and Real News Dataset (Kaggle)

The dataset contains thousands of labeled news articles categorized into:

* Fake News Articles
* Real News Articles

### Dataset Attributes

| Feature | Description         |
| ------- | ------------------- |
| title   | News Headline       |
| text    | Full News Content   |
| subject | News Category       |
| date    | Publication Date    |
| label   | Fake (0) / Real (1) |

---

# 🔄 System Workflow

```text
Dataset Collection
        ↓
Data Cleaning
        ↓
Text Preprocessing
        ↓
Feature Extraction (TF-IDF)
        ↓
Train-Test Split
        ↓
Model Training
        ↓
Performance Evaluation
        ↓
Best Model Selection
        ↓
Model Deployment
        ↓
Web Application
        ↓
News Prediction
```

---

# 🧹 Data Preprocessing

To improve model performance and eliminate noise from the dataset, the following preprocessing techniques are applied:

### Text Cleaning

* Lowercase Conversion
* URL Removal
* Punctuation Removal
* Special Character Removal
* Extra Whitespace Removal

### NLP Techniques

* Tokenization
* Stopword Removal
* Stemming
* Text Normalization

### Libraries Used

* NLTK
* re
* string
* PorterStemmer

---

# 📈 Feature Extraction

The cleaned textual data is transformed into numerical vectors using:

### TF-IDF Vectorization

```python
TfidfVectorizer(max_features=5000)
```

### Advantages

* Converts textual data into machine-readable format.
* Captures the importance of words.
* Reduces the impact of common words.
* Improves classification accuracy.

---

# 🤖 Machine Learning Models

The following classification algorithms are implemented and evaluated:

## Logistic Regression

* Fast and efficient
* Strong baseline classifier
* Suitable for text classification

## Multinomial Naive Bayes

* Designed for textual data
* Lightweight and computationally efficient

## Random Forest Classifier

* Ensemble learning technique
* Robust and highly accurate

## Passive Aggressive Classifier

* Optimized for large-scale text classification
* Frequently used in fake news detection systems

---

# 📊 Performance Evaluation

The trained models are evaluated using the following metrics:

### Accuracy

Measures overall prediction correctness.

### Precision

Measures the quality of positive predictions.

### Recall

Measures the ability to identify actual positive instances.

### F1-Score

Balances Precision and Recall.

### Confusion Matrix

Provides detailed classification insights.

---

# 📉 Data Visualization

The system generates several visual analytics including:

* Fake vs Real News Distribution
* Word Cloud for Fake News
* Word Cloud for Real News
* Top Frequent Words Analysis
* Model Accuracy Comparison Graph
* Confusion Matrix Heatmap

These visualizations help in understanding dataset patterns and model behavior.

---

# 🌐 Web Application

A Flask-based web application is developed to provide an interactive user interface.

### Prediction Workflow

```text
User Inputs News Article
            ↓
Text Preprocessing
            ↓
TF-IDF Transformation
            ↓
Model Prediction
            ↓
Confidence Score Generation
            ↓
Result Display
```

### Output Example

Input News:

"Government announces major economic reforms to boost employment."

Prediction:

✅ Real News

Confidence Score: 96.8%

---

# 💾 Database Integration

SQLite database stores prediction records for future analysis.

### Stored Information

| Field            | Description      |
| ---------------- | ---------------- |
| ID               | Unique Record ID |
| News Text        | User Input News  |
| Prediction       | Fake / Real      |
| Confidence Score | Model Confidence |
| Timestamp        | Date & Time      |

---

# 📦 Installation & Setup

### Clone Repository

```bash
git clone https://github.com/yourusername/FakeNewsDetection.git
```

### Navigate to Project Directory

```bash
cd FakeNewsDetection
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train Model

```bash
python train.py
```

### Launch Web Application

```bash
python app.py
```

### Open Browser

```text
http://127.0.0.1:5000
```

---

# 📋 Requirements

```text
numpy
pandas
scikit-learn
nltk
matplotlib
seaborn
flask
joblib
wordcloud
sqlite3
```

---

# 📸 Project Screenshots

Include the following screenshots:

* Home Page
* Prediction Result Page
* Dataset Visualization
* Word Cloud Analysis
* Accuracy Comparison Graph
* Confusion Matrix
* Admin Dashboard

---

# 🔮 Future Scope

### Advanced Deep Learning Models

* LSTM
* GRU
* BERT
* RoBERTa
* Transformers

### Future Enhancements

* Multi-Language News Detection
* Real-Time News Verification
* Browser Extension Integration
* Mobile Application Development
* Social Media Monitoring
* Live News API Integration
* Explainable AI Dashboard

---

# 📚 Learning Outcomes

This project provides practical experience in:

* Machine Learning Pipeline Development
* Natural Language Processing
* Text Classification Techniques
* Data Cleaning & Feature Engineering
* Model Evaluation & Optimization
* Flask Web Development
* Database Integration
* Data Visualization

---

# 👨‍💻 Author

**Hari Om**

Master of Computer Applications (MCA)

Machine Learning | Artificial Intelligence | Web Development

---

# 📜 License

This project is developed for educational, research, and academic purposes.

Feel free to use, modify, and enhance the project for learning and innovation.

---

# ⭐ Support

If you found this project useful, please consider giving it a **Star ⭐** on GitHub and sharing it with others interested in Machine Learning, NLP, and Fake News Detection.
