# Baseline Models Evaluation Report (Validation Set)

## 1. Summary Comparison

| Model | Class 1 (Unfair) Precision | Class 1 (Unfair) Recall | Class 1 (Unfair) F1 | PR-AUC | TN | FP | FN | TP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Keyword Rule Baseline** | 0.2500 | 0.4043 | 0.3090 | 0.1613 | 1766 | 279 | 137 | 93 |
| **TF-IDF + Logistic Regression** | 0.6156 | 0.8565 | 0.7164 | 0.7923 | 1922 | 123 | 33 | 197 |

*(Validation Set: 2,275 clauses total, 230 positive/unfair, 2,045 negative/fair).*

---

## 2. Key Observations

1. **Keyword Rule Baseline**:
   - High Recall (40.4%), capturing many clauses containing discretionary terms like "sole discretion" and "at any time".
   - Moderate Precision (25.0%), yielding 279 False Positives because benign clauses often include words like "terminate" (e.g., "you may terminate your account at any time").
   - F1 score: **0.3090**, PR-AUC: **0.1613**.

2. **TF-IDF + Logistic Regression Baseline**:
   - Logistic regression balances statistical n-gram weights with balanced class weighting.
   - Precision: **61.6%**, Recall: **85.7%**, F1 score: **0.7164**.
   - PR-AUC: **0.7923** (continuous probabilistic ranking).
   - Serves as the primary benchmark that our PyTorch ANN must compete with in Phase 3 & 4.
