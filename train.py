import os
import re
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Set styles for plots
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [10, 6]

# Define directories
DATASET_DIR = "dataset"
MODELS_DIR = "models"
PLOTS_DIR = os.path.join("static", "plots")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Step 0: Ensure NLTK packages are downloaded
print("Downloading NLTK resources...")
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception as e:
    print(f"Error downloading NLTK packages: {e}")

# Initialize Stemmer and Stopwords
stemmer = PorterStemmer()
try:
    stop_words = set(stopwords.words('english'))
except Exception:
    stop_words = set()

# Text Cleaning Function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase conversion
    text = text.lower()
    
    # 2. Remove URLs
    text = re.sub(r'https?://\s*\S+|www\.\S+', '', text)
    
    # 3. Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # 4. Remove punctuation and special characters
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 5. Remove numbers or extra spaces
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 6. Tokenization and removing stopwords and stemming
    words = word_tokenize(text)
    cleaned_words = [stemmer.stem(word) for word in words if word not in stop_words]
    
    return " ".join(cleaned_words)

def main():
    fake_path = os.path.join(DATASET_DIR, "Fake.csv")
    true_path = os.path.join(DATASET_DIR, "True.csv")
    
    # Check if files exist, generate if not
    if not os.path.exists(fake_path) or not os.path.exists(true_path):
        print("Dataset files not found. Generating mock datasets...")
        import generate_mock_data
        # This will run the generator
    
    print("Loading datasets...")
    fake = pd.read_csv(fake_path)
    true = pd.read_csv(true_path)
    
    # Add Labels
    fake["label"] = 0
    true["label"] = 1
    
    # Merge Dataset
    print("Merging datasets...")
    data = pd.concat([fake, true], ignore_index=True)
    
    # Remove Null Values
    data.dropna(subset=['text', 'label'], inplace=True)
    
    # Remove Duplicate Records
    duplicates_count = data.duplicated(subset=['text']).sum()
    print(f"Removing {duplicates_count} duplicate records...")
    data.drop_duplicates(subset=['text'], inplace=True)
    
    # Shuffle Dataset
    print("Shuffling dataset...")
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Generate distribution chart of Fake vs Real News
    print("Generating distribution chart...")
    plt.figure(figsize=(6, 4))
    counts = data['label'].value_counts()
    sns.barplot(x=['Fake News (0)', 'Real News (1)'], y=counts.values, hue=['Fake News (0)', 'Real News (1)'], palette="viridis", legend=False)
    plt.title("Distribution of Fake vs Real News")
    plt.ylabel("Number of Articles")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "correlation.png"))
    plt.close()
    
    # Generate Word Clouds
    try:
        from wordcloud import WordCloud
        print("Generating word clouds...")
        
        # Word cloud for Fake news
        fake_text = " ".join(data[data['label'] == 0]['text'].astype(str).tolist()[:50])
        wordcloud_fake = WordCloud(width=800, height=400, background_color='black', 
                                   colormap='Oranges', max_words=100).generate(fake_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_fake, interpolation='bilinear')
        plt.axis('off')
        plt.title("Word Cloud for Fake News")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "wordcloud_fake.png"))
        plt.close()

        # Word cloud for Real news
        real_text = " ".join(data[data['label'] == 1]['text'].astype(str).tolist()[:50])
        wordcloud_true = WordCloud(width=800, height=400, background_color='black', 
                                   colormap='Blues', max_words=100).generate(real_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_true, interpolation='bilinear')
        plt.axis('off')
        plt.title("Word Cloud for Real News")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "wordcloud_true.png"))
        plt.close()
    except Exception as e:
        print(f"Skipping word cloud generation: {e}")

    # Keep Only text and label
    data = data[['text', 'label']]
    
    # Text Cleaning
    print("Cleaning news text (this may take a few moments)...")
    data['cleaned_text'] = data['text'].apply(clean_text)
    
    # Remove rows where cleaned text is empty
    data = data[data['cleaned_text'] != ""]
    
    # Feature Extraction
    print("Vectorizing text using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(data['cleaned_text'])
    y = data['label'].values
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    # Define models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Passive Aggressive Classifier": PassiveAggressiveClassifier(max_iter=1000, random_state=42)
    }
    
    results = {}
    trained_model_objs = {}
    
    # Model Training and Evaluation
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        trained_model_objs[name] = model
        
        print(f"--- {name} Results ---")
        print(f"Accuracy: {acc:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("-" * 30)

    # Create Accuracy Comparison Chart
    print("Plotting model accuracy comparison...")
    plt.figure(figsize=(10, 6))
    names = list(results.keys())
    accuracies = list(results.values())
    
    # We will overwrite correlation.png to show model accuracies comparison as requested by requirements
    bars = sns.barplot(x=names, y=accuracies, hue=names, palette="muted", legend=False)
    plt.title("Model Accuracy Comparison", fontsize=14)
    plt.ylabel("Accuracy Score", fontsize=12)
    plt.ylim(0, 1.05)
    
    # Add values on top of bars
    for bar in bars.patches:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval*100:.2f}%", ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "correlation.png"))
    plt.close()
    
    # Selection of Best Model
    best_model_name = max(results, key=results.get)
    best_accuracy = results[best_model_name]
    best_model = trained_model_objs[best_model_name]
    
    print(f"\nBest Model Selected: {best_model_name} with Accuracy of {best_accuracy*100:.2f}%")
    
    # Save best model and vectorizer
    model_path = os.path.join(MODELS_DIR, "fake_news_model.pkl")
    vectorizer_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    
    joblib.dump(best_model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Saved best model to {model_path}")
    print(f"Saved TF-IDF Vectorizer to {vectorizer_path}")
    
    # Generate Confusion Matrix Heatmap for the Best Model
    print("Generating Confusion Matrix Heatmap for best model...")
    y_pred_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Fake (0)', 'Real (1)'], 
                yticklabels=['Fake (0)', 'Real (1)'])
    plt.title(f"Confusion Matrix: {best_model_name}")
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"))
    plt.close()
    
    # Also plot top 20 frequent words in dataset
    try:
        print("Generating top words plot...")
        # Get feature names from vectorizer
        feature_names = vectorizer.get_feature_names_out()
        # Sum word counts across all documents
        sum_words = X.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)[:20]
        
        df_words = pd.DataFrame(words_freq, columns=['Word', 'Frequency'])
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Frequency', y='Word', data=df_words, hue='Word', palette='rocket', legend=False)
        plt.title("Top 20 Most Frequent Words (Stemmed & Cleaned)")
        plt.xlabel("Cumulative TF-IDF Score")
        plt.ylabel("Words")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "top_words.png"))
        plt.close()
    except Exception as e:
        print(f"Skipping top words plot: {e}")
        
    print("Training pipeline finished successfully!")

if __name__ == "__main__":
    main()
