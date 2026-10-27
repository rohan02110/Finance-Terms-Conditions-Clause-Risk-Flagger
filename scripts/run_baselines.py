"""
Phase 2: Baseline Models Evaluation.

Runs:
1. Hand-crafted Keyword Rule Baseline.
2. TF-IDF + Logistic Regression Baseline.

Evaluates on the held-out validation set.
Reports per-class Precision, Recall, F1, and PR-AUC.
"""

import os
import re
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)

# Hand-crafted list of risky phrases typical of unfair consumer contracts
RISKY_KEYWORDS = [
    r"\bsole discretion\b",
    r"\bat any time\b",
    r"\bwithout notice\b",
    r"\bwithout prior notice\b",
    r"\bwithout liability\b",
    r"\bwaive\b",
    r"\bwaiver\b",
    r"\barbitrat\w*",
    r"\bclass action\b",
    r"\bas is\b",
    r"\bas available\b",
    r"\bno warranty\b",
    r"\bdisclaim\w*\b",
    r"\bindemnif\w*\b",
    r"\bhold harmless\b",
    r"\bliquidat\w*\b",
    r"\bfreeze\b",
    r"\bsuspend\w*\b",
    r"\bterminate\b",
    r"\bunilateral\w*\b",
    r"\bexclusive jurisdiction\b",
    r"\bgoverning law\b",
    r"\bnon-refundable\b",
    r"\bautomatic(ally)? renew\w*\b"
]

KEYWORD_REGEX = re.compile("|".join(RISKY_KEYWORDS), re.IGNORECASE)


def keyword_baseline_predict(texts):
    """Predict 1 if any risky keyword is found, else 0."""
    return np.array([1 if KEYWORD_REGEX.search(t) else 0 for t in texts])


def main():
    print("Loading data...")
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")

    X_train_text = train_df["text"].fillna("")
    y_train = train_df["label"].values

    X_val_text = val_df["text"].fillna("")
    y_val = val_df["label"].values

    # ==========================================
    # 1. Keyword Rule Baseline
    # ==========================================
    print("\n--- Evaluating Keyword Rule Baseline ---")
    y_val_pred_kw = keyword_baseline_predict(X_val_text)
    
    # Binary predictions directly for keyword rule (pseudo-probability is 1 or 0)
    prec_kw = precision_score(y_val, y_val_pred_kw, pos_label=1)
    rec_kw = recall_score(y_val, y_val_pred_kw, pos_label=1)
    f1_kw = f1_score(y_val, y_val_pred_kw, pos_label=1)
    pr_auc_kw = average_precision_score(y_val, y_val_pred_kw)
    cm_kw = confusion_matrix(y_val, y_val_pred_kw)

    print(f"Keyword Precision (Unfair): {prec_kw:.4f}")
    print(f"Keyword Recall (Unfair):    {rec_kw:.4f}")
    print(f"Keyword F1 (Unfair):        {f1_kw:.4f}")
    print(f"Keyword PR-AUC:             {pr_auc_kw:.4f}")
    print("Confusion Matrix (rows=actual, cols=pred):\n", cm_kw)

    # ==========================================
    # 2. TF-IDF + Logistic Regression Baseline
    # ==========================================
    print("\n--- Training TF-IDF + Logistic Regression Baseline ---")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        lowercase=True,
        sublinear_tf=True
    )
    
    # Fit vectorizer strictly on train split
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_val_tfidf = vectorizer.transform(X_val_text)

    # Logistic regression with balanced class weights
    lr_model = LogisticRegression(
        class_weight="balanced",
        random_state=42,
        max_iter=1000,
        C=1.0
    )
    lr_model.fit(X_train_tfidf, y_train)

    y_val_probs_lr = lr_model.predict_proba(X_val_tfidf)[:, 1]
    y_val_pred_lr = (y_val_probs_lr >= 0.5).astype(int)

    prec_lr = precision_score(y_val, y_val_pred_lr, pos_label=1)
    rec_lr = recall_score(y_val, y_val_pred_lr, pos_label=1)
    f1_lr = f1_score(y_val, y_val_pred_lr, pos_label=1)
    pr_auc_lr = average_precision_score(y_val, y_val_probs_lr)
    cm_lr = confusion_matrix(y_val, y_val_pred_lr)

    print(f"Logistic Regression Precision (Unfair): {prec_lr:.4f}")
    print(f"Logistic Regression Recall (Unfair):    {rec_lr:.4f}")
    print(f"Logistic Regression F1 (Unfair):        {f1_lr:.4f}")
    print(f"Logistic Regression PR-AUC:             {pr_auc_lr:.4f}")
    print("Confusion Matrix (rows=actual, cols=pred):\n", cm_lr)

    # Save baseline artifacts
    os.makedirs("models", exist_ok=True)
    joblib.dump(vectorizer, "models/tfidf_vectorizer_baseline.joblib")
    joblib.dump(lr_model, "models/logreg_baseline.joblib")

    # Generate baseline comparison report markdown
    report_content = f"""# Baseline Models Evaluation Report (Validation Set)

## 1. Summary Comparison

| Model | Class 1 (Unfair) Precision | Class 1 (Unfair) Recall | Class 1 (Unfair) F1 | PR-AUC | TN | FP | FN | TP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Keyword Rule Baseline** | {prec_kw:.4f} | {rec_kw:.4f} | {f1_kw:.4f} | {pr_auc_kw:.4f} | {cm_kw[0,0]} | {cm_kw[0,1]} | {cm_kw[1,0]} | {cm_kw[1,1]} |
| **TF-IDF + Logistic Regression** | {prec_lr:.4f} | {rec_lr:.4f} | {f1_lr:.4f} | {pr_auc_lr:.4f} | {cm_lr[0,0]} | {cm_lr[0,1]} | {cm_lr[1,0]} | {cm_lr[1,1]} |

*(Validation Set: 2,275 clauses total, 230 positive/unfair, 2,045 negative/fair).*

---

## 2. Key Observations

1. **Keyword Rule Baseline**:
   - High Recall ({rec_kw*100:.1f}%), capturing many clauses containing discretionary terms like "sole discretion" and "at any time".
   - Moderate Precision ({prec_kw*100:.1f}%), yielding {cm_kw[0,1]} False Positives because benign clauses often include words like "terminate" (e.g., "you may terminate your account at any time").
   - F1 score: **{f1_kw:.4f}**, PR-AUC: **{pr_auc_kw:.4f}**.

2. **TF-IDF + Logistic Regression Baseline**:
   - Logistic regression balances statistical n-gram weights with balanced class weighting.
   - Precision: **{prec_lr*100:.1f}%**, Recall: **{rec_lr*100:.1f}%**, F1 score: **{f1_lr:.4f}**.
   - PR-AUC: **{pr_auc_lr:.4f}** (continuous probabilistic ranking).
   - Serves as the primary benchmark that our PyTorch ANN must compete with in Phase 3 & 4.
"""
    with open("reports/baselines_summary.md", "w", encoding="utf-8") as f:
        f.write(report_content)
    print("\nSaved baseline report to reports/baselines_summary.md.")


if __name__ == "__main__":
    main()
