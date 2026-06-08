import os
import sqlite3
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, url_for

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

app = Flask(__name__)

DB_PATH = "predictions.db"
MODEL_PATH = os.path.join("models", "fake_news_model.pkl")
VECTORIZER_PATH = os.path.join("models", "tfidf_vectorizer.pkl")

# Initialize database
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            news_text TEXT NOT NULL,
            prediction INTEGER NOT NULL, -- 0 for Fake, 1 for Real
            probability REAL NOT NULL,
            credibility_score REAL NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# Load model and vectorizer
model = None
vectorizer = None

def load_ml_components():
    global model, vectorizer
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            vectorizer = joblib.load(VECTORIZER_PATH)
            print("Model and vectorizer loaded successfully.")
        except Exception as e:
            print(f"Error loading models: {e}")
    else:
        print("Warning: Model files not found. Run training script first.")

load_ml_components()

def calculate_credibility(prob, label):
    """
    Returns credibility status and percentage.
    """
    credibility = prob if label == 1 else (1.0 - prob)
    cred_pct = credibility * 100
    
    if cred_pct >= 67:
        status = "High Credibility"
        status_color = "success"
    elif cred_pct >= 34:
        status = "Medium Credibility"
        status_color = "warning"
    else:
        status = "Low Credibility"
        status_color = "danger"
        
    return status, cred_pct, status_color

@app.route('/')
def index():
    # Make sure models are loaded, reload if they weren't before
    if model is None or vectorizer is None:
        load_ml_components()
    return render_template('index.html', model_available=(model is not None))

@app.route('/predict', methods=['POST'])
def predict():
    global model, vectorizer
    if model is None or vectorizer is None:
        load_ml_components()
        if model is None or vectorizer is None:
            return jsonify({'success': False, 'error': 'Model weights not trained yet. Run train.py first.'}), 500

    data = request.get_json()
    if not data or 'text' not in data or not data['text'].strip():
        return jsonify({'success': False, 'error': 'Empty input text.'}), 400

    news_text = data['text']
    
    try:
        # Preprocess
        cleaned = clean_text(news_text)
        if not cleaned:
            # If text cleaning removes everything (e.g. just symbols), return neutral prediction
            return jsonify({
                'success': True,
                'prediction': 'Fake News',
                'probability': 50.0,
                'credibility': 50.0,
                'credibility_status': 'Low Credibility',
                'status_color': 'danger'
            })
            
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
            decision_score = model.decision_function(vectorized)[0]
            prob = 1 / (1 + np.exp(-decision_score))
            
        confidence = prob if pred == 1 else (1.0 - prob)
        confidence_pct = confidence * 100
        
        # Calculate credibility
        cred_status, cred_pct, status_color = calculate_credibility(prob, pred)
        
        # Save to database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO history (news_text, prediction, probability, credibility_score)
            VALUES (?, ?, ?, ?)
        ''', (news_text, pred, confidence_pct, cred_pct))
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'prediction': 'Real News' if pred == 1 else 'Fake News',
            'probability': round(confidence_pct, 2),
            'credibility': round(cred_pct, 2),
            'credibility_status': cred_status,
            'status_color': status_color
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/dashboard')
def dashboard():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get total count
    cursor.execute('SELECT COUNT(*) FROM history')
    total_predictions = cursor.fetchone()[0]
    
    # Get Fake News count (prediction = 0)
    cursor.execute('SELECT COUNT(*) FROM history WHERE prediction = 0')
    fake_count = cursor.fetchone()[0]
    
    # Get Real News count (prediction = 1)
    cursor.execute('SELECT COUNT(*) FROM history WHERE prediction = 1')
    real_count = cursor.fetchone()[0]
    
    # Get history list (limit to 50 latest)
    cursor.execute('''
        SELECT id, news_text, prediction, probability, credibility_score, timestamp 
        FROM history 
        ORDER BY timestamp DESC 
        LIMIT 50
    ''')
    raw_history = cursor.fetchall()
    
    # Format history records
    history = []
    for row in raw_history:
        # Truncate text preview
        preview = row[1][:100] + "..." if len(row[1]) > 100 else row[1]
        
        # Check label
        pred_label = "Real News" if row[2] == 1 else "Fake News"
        badge_class = "success" if row[2] == 1 else "danger"
        
        # Calculate credibility status
        # Since DB stores confidence and credibility directly
        cred_score = row[4]
        if cred_score >= 67:
            cred_status = "High"
            cred_badge = "success"
        elif cred_score >= 34:
            cred_status = "Medium"
            cred_badge = "warning"
        else:
            cred_status = "Low"
            cred_badge = "danger"
            
        history.append({
            'id': row[0],
            'text_preview': preview,
            'prediction': pred_label,
            'badge_class': badge_class,
            'probability': round(row[3], 1),
            'credibility': round(cred_score, 1),
            'credibility_status': cred_status,
            'cred_badge': cred_badge,
            'timestamp': row[5]
        })
        
    conn.close()
    
    # Check what charts are available in static/plots
    plots = {
        'correlation': os.path.exists(os.path.join('static', 'plots', 'correlation.png')),
        'confusion_matrix': os.path.exists(os.path.join('static', 'plots', 'confusion_matrix.png')),
        'wordcloud_fake': os.path.exists(os.path.join('static', 'plots', 'wordcloud_fake.png')),
        'wordcloud_true': os.path.exists(os.path.join('static', 'plots', 'wordcloud_true.png')),
        'top_words': os.path.exists(os.path.join('static', 'plots', 'top_words.png'))
    }
    
    # Active classifier
    active_classifier = "No model loaded"
    if model is not None:
        active_classifier = type(model).__name__
        
    return render_template('dashboard.html', 
                           total=total_predictions, 
                           fake=fake_count, 
                           real=real_count, 
                           history=history,
                           plots=plots,
                           classifier=active_classifier)

@app.route('/clear_history', methods=['POST'])
def clear_history():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM history')
    conn.commit()
    conn.close()
    return redirect(url_for('dashboard'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
