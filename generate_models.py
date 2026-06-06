"""
Run this script once inside your project folder to generate all 4 model files.
Make sure your venv is activated and UpdatedResumeDataSet.csv is in the same folder.

    python generate_models.py
"""

import os
import re
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# ── helpers ──────────────────────────────────────────────────────────────────

def clean_resume(txt):
    txt = re.sub(r'http\S+\s', ' ', txt)
    txt = re.sub(r'RT|cc', ' ', txt)
    txt = re.sub(r'#\S+\s', ' ', txt)
    txt = re.sub(r'@\S+', ' ', txt)
    txt = re.sub(r'[%s]' % re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', txt)
    txt = re.sub(r'[^\x00-\x7f]', ' ', txt)
    txt = re.sub(r'\s+', ' ', txt)
    return txt.strip()

# ── load dataset ─────────────────────────────────────────────────────────────

csv_path = "UpdatedResumeDataSet.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(
        f"\n'{csv_path}' not found.\n"
        "Make sure UpdatedResumeDataSet.csv is in the same folder as this script."
    )

print("Loading dataset...")
df = pd.read_csv(csv_path)
print(f"  Loaded {len(df)} rows, columns: {list(df.columns)}")

# Normalise column names – the CSV uses 'Resume' and 'Category'
df.columns = [c.strip() for c in df.columns]
df['Resume'] = df['Resume'].astype(str).apply(clean_resume)

# ── MODEL 1 & 2 : categorization ─────────────────────────────────────────────

print("\nTraining categorization model...")
X = df['Resume']
y = df['Category']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

tfidf_cat = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_cat.fit_transform(X_train)
X_test_tfidf  = tfidf_cat.transform(X_test)

rf_cat = RandomForestClassifier(n_estimators=100, random_state=42)
rf_cat.fit(X_train_tfidf, y_train)

acc = rf_cat.score(X_test_tfidf, y_test)
print(f"  Categorization accuracy: {acc:.2%}")

# ── MODEL 3 & 4 : job recommendation ─────────────────────────────────────────
# jobs_dataset_with_features.csv is missing from the repo, so we reuse the
# same dataset with Category as the "recommended role".  This gives the app
# a working model instead of crashing on startup.

print("\nTraining job-recommendation model (using Category as role)...")
tfidf_job = TfidfVectorizer(max_features=5000)
X_train_tfidf2 = tfidf_job.fit_transform(X_train)
X_test_tfidf2  = tfidf_job.transform(X_test)

rf_job = RandomForestClassifier(n_estimators=100, random_state=42)
rf_job.fit(X_train_tfidf2, y_train)

acc2 = rf_job.score(X_test_tfidf2, y_test)
print(f"  Job-recommendation accuracy: {acc2:.2%}")

# ── save models ───────────────────────────────────────────────────────────────

os.makedirs("models", exist_ok=True)

pickle.dump(rf_cat,   open("models/rf_classifier_categorization.pkl",    "wb"))
pickle.dump(tfidf_cat, open("models/tfidf_vectorizer_categorization.pkl", "wb"))
pickle.dump(rf_job,   open("models/rf_classifier_job_recommendation.pkl",    "wb"))
pickle.dump(tfidf_job, open("models/tfidf_vectorizer_job_recommendation.pkl", "wb"))

print("\n✅ All 4 model files saved to models/")
print("   models/rf_classifier_categorization.pkl")
print("   models/tfidf_vectorizer_categorization.pkl")
print("   models/rf_classifier_job_recommendation.pkl")
print("   models/tfidf_vectorizer_job_recommendation.pkl")
print("\nNow run:  python APP.py")
