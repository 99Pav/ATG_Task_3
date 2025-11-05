#!/usr/bin/env python3
import json, os, sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, "sample_data", "samples.json")
MODEL_DIR = os.path.join(ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def load_samples(path):
    if not os.path.exists(path):
        print(f"[ERROR] Sample data not found at {path}", file=sys.stderr)
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = [d["text"] for d in data]
    labels = [d["label"] for d in data]
    return texts, labels

def train():
    print("[TRAIN] Loading samples...")
    texts, labels = load_samples(DATA_PATH)
    if len(texts) < 2:
        print("[ERROR] Not enough samples to train.", file=sys.stderr)
        sys.exit(1)
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
    pipe = make_pipeline(TfidfVectorizer(ngram_range=(1,2), max_features=5000), LogisticRegression(max_iter=1000))
    print("[TRAIN] Fitting model...")
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"[TRAIN] Test accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))
    out_path = os.path.join(MODEL_DIR, "backup_model.joblib")
    dump(pipe, out_path)
    print(f"[TRAIN] Backup model saved to {out_path}")

if __name__ == '__main__':
    try:
        train()
    except Exception as e:
        print(f"[ERROR] Training failed: {e}", file=sys.stderr)
        raise
