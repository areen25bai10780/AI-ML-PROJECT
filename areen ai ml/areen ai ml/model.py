import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Important for server environments
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ── File Paths ──────────────────────────────────
DATA_PATH  = "data/news.csv"
VEC_PATH   = "models/vectorizer.pkl"
LR_PATH    = "models/logistic_regression.pkl"
DT_PATH    = "models/decision_tree.pkl"

def check_status():
    """Check if the models and data have been generated yet."""
    status = {
        "dataset_exists": os.path.exists(DATA_PATH),
        "models_exist": os.path.exists(VEC_PATH) and os.path.exists(LR_PATH) and os.path.exists(DT_PATH)
    }
    return status

def create_dataset():
    os.makedirs("data", exist_ok=True)
    if os.path.exists(DATA_PATH):
        return {"status": "success", "message": "Dataset already exists."}

    fake = [
        "SHOCKING: Government hiding alien technology",
        "Vaccines cause autism, media is silent about it",
        "President secretly meets with foreign spies",
        "Big pharma suppressing cancer cure for profit",
        "NASA admits moon landing was fake all along",
        "Miracle cure discovered, doctors don't want you to know",
        "Election results manipulated by deep state insiders",
        "Secret society controls all world governments",
        "Time traveler warns of coming global catastrophe",
        "Mind control signals broadcast through 5G towers",
        "Billionaires plan to reduce world population secretly",
        "Climate change is a complete hoax leaked emails show",
        "Underground cities built for elite when apocalypse comes",
        "Ancient prophecy predicts end of world next month",
        "Whistleblower exposes truth about chemtrails conspiracy",
    ]
    real = [
        "Federal Reserve raises interest rates by 25 basis points",
        "Scientists develop new treatment for Alzheimer disease",
        "Parliament passes new climate change legislation today",
        "Stock markets fall amid global economic uncertainty",
        "Researchers publish findings on COVID vaccine efficacy",
        "City council approves new affordable housing development",
        "Olympic committee announces host city for 2032 games",
        "Tech company reports quarterly earnings above expectations",
        "Study finds Mediterranean diet reduces heart disease risk",
        "New legislation aims to regulate artificial intelligence",
        "Scientists discover new dinosaur species in Argentina",
        "Government announces new infrastructure spending plan",
        "University develops more efficient solar panel technology",
        "Health ministry launches new vaccination awareness campaign",
        "Report highlights growing income inequality worldwide",
    ]

    fake_data = [fake[i % len(fake)] for i in range(300)]
    real_data = [real[i % len(real)] for i in range(300)]

    df = pd.DataFrame({
        "text":  fake_data + real_data,
        "label": ["FAKE"] * 300 + ["REAL"] * 300
    }).sample(frac=1, random_state=42).reset_index(drop=True)

    df.to_csv(DATA_PATH, index=False)
    return {"status": "success", "message": f"Dataset created with {len(df)} samples"}

def train():
    os.makedirs("models", exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df["text"] = df["text"].str.lower()
    df["label"] = df["label"].map({"FAKE": 1, "REAL": 0})

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)
    with open(VEC_PATH, "wb") as f: pickle.dump(vectorizer, f)

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_vec, y_train)
    lr_acc = accuracy_score(y_test, lr.predict(X_test_vec))
    with open(LR_PATH, "wb") as f: pickle.dump(lr, f)

    dt = DecisionTreeClassifier(max_depth=15, random_state=42)
    dt.fit(X_train_vec, y_train)
    dt_acc = accuracy_score(y_test, dt.predict(X_test_vec))
    with open(DT_PATH, "wb") as f: pickle.dump(dt, f)

    os.makedirs("static/plots", exist_ok=True)
    plt.figure(figsize=(6,4))
    plt.bar(["Logistic Regression", "Decision Tree"],
            [lr_acc * 100, dt_acc * 100],
            color=["#3498db", "#2ecc71"], edgecolor="black")
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy (%)")
    plt.ylim([0, 110])
    for i, v in enumerate([lr_acc*100, dt_acc*100]):
        plt.text(i, v + 2, f"{v:.1f}%", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig("static/plots/accuracy.png", dpi=150)
    plt.close()
    
    return {
        "status": "success", 
        "lr_accuracy": round(lr_acc*100, 2),
        "dt_accuracy": round(dt_acc*100, 2)
    }

def predict_news(text):
    if len(text.strip()) < 5:
        return {"status": "error", "message": "Text is too short."}
        
    try:
        with open(VEC_PATH, "rb") as f: vectorizer = pickle.load(f)
        with open(LR_PATH,  "rb") as f: lr = pickle.load(f)
        with open(DT_PATH,  "rb") as f: dt = pickle.load(f)
    except FileNotFoundError:
        return {"status": "error", "message": "Models not found. Please train first."}

    vec = vectorizer.transform([text.lower().strip()])
    
    # Get probabilities
    lr_prob = lr.predict_proba(vec)[0]
    dt_prob = dt.predict_proba(vec)[0]
    
    lr_pred_idx = lr.predict(vec)[0]
    dt_pred_idx = dt.predict(vec)[0]
    
    lr_pred = "FAKE" if lr_pred_idx == 1 else "REAL"
    dt_pred = "FAKE" if dt_pred_idx == 1 else "REAL"
    
    lr_conf = lr_prob[lr_pred_idx] * 100
    dt_conf = dt_prob[dt_pred_idx] * 100

    return {
        "status": "success",
        "results": {
            "logistic_regression": {
                "prediction": lr_pred,
                "confidence": round(lr_conf, 1)
            },
            "decision_tree": {
                "prediction": dt_pred,
                "confidence": round(dt_conf, 1)
            },
            # Final verdict uses LR, as it's generally more robust for text
            "final_verdict": lr_pred
        }
    }
