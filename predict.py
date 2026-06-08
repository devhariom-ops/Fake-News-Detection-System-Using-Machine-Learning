import os
import sys
import joblib
import numpy as np

# Import clean_text from train.py
try:
    from train import clean_text
except ImportError:
    # Inline fallback if train.py is not in path (unlikely)
    import re
    import string
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    from nltk.tokenize import word_tokenize
    
    stemmer = PorterStemmer()
    try:
        stop_words = set(stopwords.words('english'))
    except Exception:
        stop_words = set()
        
    def clean_text(text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'https?://\s*\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\w*\d\w*', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        words = word_tokenize(text)
        cleaned_words = [stemmer.stem(word) for word in words if word not in stop_words]
        return " ".join(cleaned_words)

def get_credibility_level(prob, label):
    """
    Returns credibility status and percentage.
    If label is 1 (Real), credibility equals probability.
    If label is 0 (Fake), credibility equals 1 - probability.
    """
    credibility = prob if label == 1 else (1.0 - prob)
    cred_pct = credibility * 100
    
    if cred_pct >= 67:
        status = "High Credibility"
    elif cred_pct >= 34:
        status = "Medium Credibility"
    else:
        status = "Low Credibility"
        
    return status, cred_pct

def main():
    model_path = os.path.join("models", "fake_news_model.pkl")
    vectorizer_path = os.path.join("models", "tfidf_vectorizer.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        print("Error: Trained model and vectorizer not found.")
        print("Please run the training pipeline first: python train.py")
        sys.exit(1)
        
    # Load model and vectorizer
    print("Loading model and vectorizer...")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    print("Model loaded successfully!")
    print(f"Active classifier: {type(model).__name__}")
    print("=" * 50)
    
    while True:
        try:
            print("\nEnter News Text (press Ctrl+C or type 'exit' to quit):")
            user_input = input("Enter News: ")
            
            if user_input.strip().lower() == 'exit':
                break
                
            if not user_input.strip():
                print("Empty input. Please enter some news text.")
                continue
                
            # Preprocess
            cleaned = clean_text(user_input)
            
            # Vectorize
            vectorized = vectorizer.transform([cleaned])
            
            # Predict
            pred = int(model.predict(vectorized)[0])
            
            # Calculate Probability
            prob = 0.5
            if hasattr(model, "predict_proba"):
                prob_arr = model.predict_proba(vectorized)[0]
                prob = prob_arr[1]  # Probability of class 1 (Real)
            elif hasattr(model, "decision_function"):
                # Fallback for PassiveAggressive or SVMs using sigmoid scaling
                decision_score = model.decision_function(vectorized)[0]
                prob = 1 / (1 + np.exp(-decision_score))
            
            # Probability score relative to prediction
            confidence = prob if pred == 1 else (1.0 - prob)
            confidence_pct = confidence * 100
            
            # Credibility Meter
            cred_status, cred_pct = get_credibility_level(prob, pred)
            
            # Output Results
            print("\n" + "-" * 40)
            print("PREDICTION RESULT:")
            if pred == 1:
                print("Prediction: Real News")
                print(f"Probability Score: {confidence_pct:.2f}% Real")
            else:
                print("Prediction: Fake News")
                print(f"Probability Score: {confidence_pct:.2f}% Fake")
                
            print(f"Credibility Meter: {cred_status} ({cred_pct:.2f}% Credibility)")
            print("-" * 40)
            
        except KeyboardInterrupt:
            print("\nExiting CLI predictor.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
