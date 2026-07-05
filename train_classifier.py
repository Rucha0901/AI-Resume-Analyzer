# train_classifier.py

import pandas as pd
import numpy as np
import re
import joblib
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Create folder to save trained model
os.makedirs("models", exist_ok=True)


# ================================================================
# SECTION 1: LOAD DATA
# ================================================================

def load_data(filepath):
    """
    Loads the CSV dataset into a pandas DataFrame.
    Drops any rows where Resume text or Category is missing.
    """
    
    print("Loading dataset...")
    df = pd.read_csv(filepath)
    
    # Drop rows with missing values
    # dropna() removes rows where ANY column has NaN (empty value)
    df = df.dropna(subset=['Resume', 'Category'])
    
    # Remove duplicate resumes (same text appearing twice)
    df = df.drop_duplicates(subset=['Resume'])
    
    print(f"Loaded {len(df)} resumes across {df['Category'].nunique()} categories")
    return df


# ================================================================
# SECTION 2: CLEAN RESUME TEXT
# ================================================================

def clean_resume_text(text):
    """
    Cleans a single resume text for ML processing.
    Same logic as Day 3-4 but slightly simplified for batch processing.
    
    We process 2484 resumes so this function runs 2484 times.
    """
    
    # Fix encoding issues
    text = text.encode('ascii', errors='ignore').decode('ascii')
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', ' ', text)
    
    # Remove phone numbers
    text = re.sub(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', ' ', text)
    
    # Remove special characters, keep letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Lowercase
    text = text.lower().strip()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered = [w for w in tokens if w not in stop_words and len(w) > 2]
    
    return " ".join(filtered)


def preprocess_dataset(df):
    """
    Runs clean_resume_text on every resume in the dataset.
    
    df['Resume'].apply(function) means:
    "Call this function on every value in the Resume column"
    It's like a loop but faster.
    """
    
    print("\nCleaning all resume texts...")
    print("(This may take 1-2 minutes for 2484 resumes...)")
    
    # Apply cleaning to every resume
    df['cleaned_resume'] = df['Resume'].apply(clean_resume_text)
    
    # Check how cleaning affected length
    original_avg = df['Resume'].apply(len).mean()
    cleaned_avg  = df['cleaned_resume'].apply(len).mean()
    
    print(f"Average length before cleaning: {original_avg:.0f} chars")
    print(f"Average length after cleaning:  {cleaned_avg:.0f} chars")
    print(f"Reduction: {((original_avg-cleaned_avg)/original_avg*100):.1f}%")
    
    return df


# ================================================================
# SECTION 3: ENCODE LABELS
# ================================================================

def encode_labels(df):
    """
    Machine learning models work with NUMBERS, not text.
    Your category labels are text: "Data Science", "HR", "Java Developer"
    
    LabelEncoder converts them to numbers:
    "Data Science"   → 0
    "DevOps Engineer"→ 1  
    "HR"             → 2
    "Java Developer" → 3
    ... and so on
    
    We SAVE the encoder so we can reverse this later:
    When model predicts "3", we look up the encoder to get "Java Developer"
    """
    
    print("\nEncoding category labels...")
    
    le = LabelEncoder()
    
    # fit_transform: learns all unique categories AND converts them
    df['label'] = le.fit_transform(df['Category'])
    
    print("Category → Number mapping:")
    for i, category in enumerate(le.classes_):
        count = (df['Category'] == category).sum()
        print(f"  {i:2d} → {category} ({count} resumes)")
    
    return df, le


# ================================================================
# SECTION 4: TF-IDF VECTORIZATION
# ================================================================

def vectorize_resumes(df):
    """
    Converts all cleaned resume texts into TF-IDF vectors.
    
    This time we're vectorizing 2484 documents (not just 2).
    The process is the same as Day 7-8, but now:
    - IDF is calculated across ALL 2484 resumes
    - Each resume becomes a vector of the same size
    - The result is a matrix: 2484 rows × vocabulary_size columns
    
    Parameters chosen carefully:
    max_features=15000 → keep top 15000 words by TF-IDF score
    ngram_range=(1,2)  → single words AND word pairs
    min_df=2           → ignore words appearing in less than 2 resumes
                         (removes typos and ultra-rare words)
    max_df=0.95        → ignore words appearing in more than 95% of resumes
                         (removes words that are so common they're useless)
    sublinear_tf=True  → log scaling (prevents very frequent words dominating)
    """
    
    print("\nVectorizing resumes with TF-IDF...")
    
    vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    
    # Fit on ALL resume texts and transform them
    X = vectorizer.fit_transform(df['cleaned_resume'])
    
    print(f"Vectorization complete!")
    print(f"Matrix shape: {X.shape}")
    print(f"  → {X.shape[0]} resumes, each represented as {X.shape[1]} features")
    
    return X, vectorizer


# ================================================================
# SECTION 5: TRAIN-TEST SPLIT
# ================================================================

def split_data(X, y):
    """
    Split data into training set and test set.
    
    WHY SPLIT?
    If you test on the same data you trained on, the model
    looks 100% accurate — but that's CHEATING. It just memorized
    the answers. We need to test on data it has NEVER seen.
    
    80% → Training set: model LEARNS from these
    20% → Test set: model is EVALUATED on these (never seen during training)
    
    Parameters:
    test_size=0.2      → 20% goes to test set
    random_state=42    → fixes the random split so results are reproducible
                         (42 is a convention, any number works)
    stratify=y         → IMPORTANT: ensures each category is proportionally
                         represented in BOTH train and test sets.
                         Without this, test set might accidentally have
                         no "Hadoop" resumes at all!
    """
    
    print("\nSplitting data into train/test sets...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} resumes")
    print(f"Test set:     {X_test.shape[0]} resumes")
    print(f"Split ratio:  {X_train.shape[0]/len(y)*100:.0f}% train / {X_test.shape[0]/len(y)*100:.0f}% test")
    
    return X_train, X_test, y_train, y_test


# ================================================================
# SECTION 6: TRAIN THE MODELS
# ================================================================

def train_logistic_regression(X_train, y_train):
    """
    Trains a Logistic Regression classifier.
    
    WHAT IS LOGISTIC REGRESSION? (Simple explanation)
    
    Despite the name, it's used for CLASSIFICATION not regression.
    
    Think of it like a voting system:
    - For each word in a resume, it has a "vote weight" per category
    - "python" has high weight for "Data Science" and "Python Developer"
    - "photoshop" has high weight for "Web Designing"
    - "recruiting" has high weight for "HR"
    
    During training, it adjusts these weights until predictions are correct.
    During prediction, it multiplies each word's TF-IDF score by its weight,
    sums everything up per category, and picks the category with highest sum.
    
    ADVANTAGES:
    - Fast to train (seconds, not minutes)
    - Easy to understand
    - Works very well for text classification
    - Gives probability scores (e.g., "72% Data Science, 15% Python Dev")
    
    Parameters:
    max_iter=1000     → maximum training iterations before stopping
                        (increase if you see "ConvergenceWarning")
    C=1.0             → regularization strength (controls overfitting)
                        higher C = model fits training data more tightly
    solver='lbfgs'    → the optimization algorithm used internally
    multi_class='auto'→ handles multiple categories automatically
    """
    
    print("\nTraining Logistic Regression model...")
    
    lr_model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver='lbfgs',
        multi_class='auto',
        random_state=42
    )
    
    # FIT = the actual training step
    # The model looks at X_train (resume vectors) and y_train (labels)
    # and adjusts its internal weights to minimize prediction errors
    lr_model.fit(X_train, y_train)
    
    print("Logistic Regression training complete!")
    return lr_model


def train_random_forest(X_train, y_train):
    """
    Trains a Random Forest classifier.
    
    WHAT IS RANDOM FOREST? (Simple explanation)
    
    Instead of one decision tree, it builds MANY trees (a forest).
    Each tree is trained on a random subset of the data and features.
    
    Think of it like asking 100 different experts and taking a vote:
    - Expert 1 (Tree 1) looks at words: python, machine, learning → votes "Data Science"
    - Expert 2 (Tree 2) looks at words: sql, analysis, python → votes "Data Science"  
    - Expert 3 (Tree 3) looks at words: java, spring, api → votes "Java Developer"
    - Final decision: majority vote wins → "Data Science"
    
    The "random" part: each tree sees different random words and
    random resumes, so they make different mistakes. Combined,
    their errors cancel out — this is called "ensemble learning".
    
    ADVANTAGES:
    - Very robust (hard to fool)
    - Handles complex patterns well
    - Gives feature importances (which words matter most?)
    
    DISADVANTAGE:
    - Slower to train than Logistic Regression
    - Uses more memory
    
    Parameters:
    n_estimators=200   → build 200 trees (more = better but slower)
    max_depth=None     → trees can grow as deep as needed
    min_samples_split=5→ need at least 5 samples to split a node
                         (prevents overfitting)
    random_state=42    → for reproducibility
    n_jobs=-1          → use ALL available CPU cores for speed
    """
    
    print("\nTraining Random Forest model...")
    print("(This takes 2-5 minutes, please wait...)")
    
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    
    print("Random Forest training complete!")
    return rf_model


# ================================================================
# SECTION 7: EVALUATE THE MODELS
# ================================================================

def evaluate_model(model, X_test, y_test, label_encoder, model_name):
    """
    Tests the trained model on data it has NEVER seen (test set).
    Calculates multiple performance metrics.
    
    METRICS EXPLAINED:
    
    ACCURACY:
    Simply: what % of predictions were correct?
    accuracy = correct predictions / total predictions
    Example: 180 correct out of 200 test resumes = 90% accuracy
    
    Limitation: if 90% of your data is "Java Developer",
    a model that always predicts "Java Developer" gets 90% accuracy
    but is totally useless. That's why we also look at F1-score.
    
    PRECISION (per category):
    Of all resumes the model PREDICTED as "Data Science",
    what fraction actually WERE Data Science?
    Example: model predicted 20 resumes as Data Science,
    but only 16 actually were → precision = 16/20 = 80%
    
    RECALL (per category):
    Of all resumes that ACTUALLY ARE "Data Science",
    what fraction did the model correctly identify?
    Example: there were 25 actual Data Science resumes,
    model found 16 of them → recall = 16/25 = 64%
    
    F1-SCORE:
    The harmonic mean of Precision and Recall.
    F1 = 2 × (Precision × Recall) / (Precision + Recall)
    This single number balances both — use this as your main metric.
    
    CONFUSION MATRIX:
    A table showing which categories got confused with which.
    Rows = actual categories
    Columns = predicted categories
    Diagonal = correct predictions
    Off-diagonal = mistakes (e.g., Python Dev predicted as Data Science)
    """
    
    print(f"\n{'='*55}")
    print(f"  EVALUATION: {model_name}")
    print(f"{'='*55}")
    
    # Make predictions on test set
    y_pred = model.predict(X_test)
    
    # ── METRIC 1: ACCURACY ─────────────────────────────
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy*100:.2f}%")
    print(f"(Correctly classified {int(accuracy*len(y_test))} out of {len(y_test)} test resumes)")
    
    # ── METRIC 2: F1-SCORE ─────────────────────────────
    # weighted = accounts for how many resumes are in each category
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    f1_macro    = f1_score(y_test, y_pred, average='macro')
    
    print(f"\nF1-Score (weighted): {f1_weighted*100:.2f}%")
    print(f"F1-Score (macro):    {f1_macro*100:.2f}%")
    print("  weighted = accounts for class imbalance (recommended)")
    print("  macro    = treats all classes equally")
    
    # ── METRIC 3: CLASSIFICATION REPORT ────────────────
    # Shows precision, recall, F1 for EVERY category
    print(f"\nDetailed Classification Report:")
    category_names = label_encoder.classes_
    report = classification_report(
        y_test, y_pred,
        target_names=category_names
    )
    print(report)
    
    # ── METRIC 4: CONFUSION MATRIX ─────────────────────
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\nConfusion Matrix (abbreviated — showing mistakes):")
    print("(Diagonal = correct predictions)")
    
    # Find the worst-confused pairs
    cm_copy = cm.copy()
    np.fill_diagonal(cm_copy, 0)  # zero out diagonal (correct predictions)
    
    # Find top 5 most confused pairs
    print("\nTop confusion pairs (most common mistakes):")
    for _ in range(5):
        idx = np.unravel_index(cm_copy.argmax(), cm_copy.shape)
        if cm_copy[idx] > 0:
            actual    = category_names[idx[0]]
            predicted = category_names[idx[1]]
            count     = cm_copy[idx]
            print(f"  '{actual}' mistaken as '{predicted}': {count} times")
            cm_copy[idx] = 0
    
    return {
        "accuracy"   : accuracy,
        "f1_weighted": f1_weighted,
        "f1_macro"   : f1_macro,
        "y_pred"     : y_pred,
        "cm"         : cm
    }


# ================================================================
# SECTION 8: SAVE THE MODELS
# ================================================================

def save_models(lr_model, rf_model, vectorizer, label_encoder):
    """
    Saves trained models to disk using joblib.
    
    WHY SAVE MODELS?
    Training takes minutes. You don't want to retrain every time
    you start your app. Save once, load instantly forever.
    
    joblib is better than pickle for sklearn models because
    it handles large numpy arrays more efficiently.
    
    What we save:
    1. The trained Logistic Regression model
    2. The trained Random Forest model
    3. The TF-IDF vectorizer (CRITICAL: must use SAME vectorizer
       that was fitted during training — same vocabulary, same IDF values)
    4. The Label Encoder (to convert numbers back to category names)
    """
    
    print("\nSaving models to disk...")
    
    joblib.dump(lr_model,      "models/logistic_regression_model.pkl")
    joblib.dump(rf_model,      "models/random_forest_model.pkl")
    joblib.dump(vectorizer,    "models/tfidf_vectorizer.pkl")
    joblib.dump(label_encoder, "models/label_encoder.pkl")
    
    print("Models saved successfully:")
    print("  models/logistic_regression_model.pkl")
    print("  models/random_forest_model.pkl")
    print("  models/tfidf_vectorizer.pkl")
    print("  models/label_encoder.pkl")


# ================================================================
# SECTION 9: PREDICT ON NEW RESUME
# ================================================================

def predict_resume_category(resume_text, model, vectorizer, label_encoder):
    """
    Takes a NEW resume text (never seen during training),
    cleans it, vectorizes it, and predicts its job category.
    
    This is what your Streamlit app will call.
    """
    
    # Step 1: Clean the text (same cleaning as training data)
    cleaned = clean_resume_text(resume_text)
    
    # Step 2: Vectorize using the SAME fitted vectorizer from training
    # IMPORTANT: use transform() NOT fit_transform()
    # fit_transform() would rebuild the vocabulary — wrong!
    # transform() uses the vocabulary learned during training — correct!
    vector = vectorizer.transform([cleaned])
    
    # Step 3: Predict category number
    prediction_number = model.predict(vector)[0]
    
    # Step 4: Get probability scores for ALL categories
    probabilities = model.predict_proba(vector)[0]
    
    # Step 5: Convert number back to category name
    predicted_category = label_encoder.inverse_transform([prediction_number])[0]
    confidence = probabilities[prediction_number] * 100
    
    # Get top 3 predictions with their probabilities
    top3_indices = np.argsort(probabilities)[-3:][::-1]
    top3 = [
        {
            "category"   : label_encoder.inverse_transform([i])[0],
            "probability": round(probabilities[i] * 100, 2)
        }
        for i in top3_indices
    ]
    
    return {
        "predicted_category": predicted_category,
        "confidence"        : round(confidence, 2),
        "top3_predictions"  : top3
    }


# ================================================================
# SECTION 10: RUN EVERYTHING
# ================================================================

if __name__ == "__main__":
    
    # ── STEP 1: Load data ──────────────────────────────
    df = load_data("data/resume_dataset.csv")
    
    # ── STEP 2: Preprocess ────────────────────────────
    df = preprocess_dataset(df)
    
    # ── STEP 3: Encode labels ─────────────────────────
    df, label_encoder = encode_labels(df)
    
    # ── STEP 4: Vectorize ─────────────────────────────
    X, vectorizer = vectorize_resumes(df)
    y = df['label'].values
    
    # ── STEP 5: Split data ────────────────────────────
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # ── STEP 6: Train both models ─────────────────────
    lr_model = train_logistic_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    
    # ── STEP 7: Evaluate both models ──────────────────
    lr_results = evaluate_model(lr_model, X_test, y_test, label_encoder, "Logistic Regression")
    rf_results = evaluate_model(rf_model, X_test, y_test, label_encoder, "Random Forest")
    
    # ── STEP 8: Compare and pick winner ───────────────
    print("\n" + "="*55)
    print("  MODEL COMPARISON")
    print("="*55)
    print(f"\nLogistic Regression → Accuracy: {lr_results['accuracy']*100:.2f}%  F1: {lr_results['f1_weighted']*100:.2f}%")
    print(f"Random Forest       → Accuracy: {rf_results['accuracy']*100:.2f}%  F1: {rf_results['f1_weighted']*100:.2f}%")
    
    if lr_results['f1_weighted'] >= rf_results['f1_weighted']:
        print("\nWinner: Logistic Regression (use this as your primary model)")
        best_model = lr_model
    else:
        print("\nWinner: Random Forest (use this as your primary model)")
        best_model = rf_model
    
    # ── STEP 9: Save everything ───────────────────────
    save_models(lr_model, rf_model, vectorizer, label_encoder)
    
    # ── STEP 10: Test on a brand new resume ───────────
    print("\n" + "="*55)
    print("  TESTING ON A BRAND NEW RESUME")
    print("="*55)
    
    test_resume = """
    Experienced Data Scientist with 4 years of experience.
    Proficient in Python, Machine Learning, Deep Learning, TensorFlow.
    Strong knowledge of SQL, Pandas, NumPy, data analysis and visualization.
    Experience with AWS cloud platform and Docker containerization.
    Published 2 research papers on NLP and computer vision.
    """
    
    result = predict_resume_category(
        test_resume,
        best_model,
        vectorizer,
        label_encoder
    )
    
    print(f"\nInput resume summary: Data Scientist resume")
    print(f"Predicted Category: {result['predicted_category']}")
    print(f"Confidence: {result['confidence']}%")
    print(f"\nTop 3 predictions:")
    for i, pred in enumerate(result['top3_predictions'], 1):
        bar = "█" * int(pred['probability'] / 5)
        print(f"  {i}. {pred['category']:<25} {pred['probability']:6.2f}% {bar}")