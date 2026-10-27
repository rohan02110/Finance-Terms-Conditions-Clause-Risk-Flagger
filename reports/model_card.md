# Model Card: Finance Terms & Conditions Clause Risk Flagger

## 1. Model Details
- **Architecture**: Feed-Forward Artificial Neural Network (Multilayer Perceptron / MLP).
  - Layer 1: `Linear(5000, 256)` + `ReLU` + `Dropout(0.3)`
  - Layer 2: `Linear(256, 64)` + `ReLU`
  - Output Layer: `Linear(64, 1)` + Sigmoid probability calibration.
- **Input Features**: 5,000 sublinear TF-IDF features (unigrams and bigrams, strictly fitted on training data).
- **Optimization**: Adam (`lr = 0.001`), `BCEWithLogitsLoss` with positive class weighting (`pos_weight = 7.78`) to counter severe class imbalance.
- **Framework**: PyTorch 2.x, scikit-learn, Python 3.11+.

---

## 2. Intended Use
- **Primary Use Case**: Educational decision-support tool to highlight potentially unfair, one-sided, or restrictive contractual clauses in finance and investment Terms & Conditions.
- **Target Users**: Consumers, students, and retail investors reading terms for fintech, trading, neo-banking, and digital wallet applications.
- **Out of Scope & Prohibited Uses**:
  - **Not Legal or Financial Advice**: This model must **never** be used as a substitute for professional legal review.
  - **No Safety Guarantee**: The system must **never** certify an agreement as "safe to sign", "low risk", or legally binding.

---

## 3. Training & Evaluation Data
- **Corpus**: CLAUDETTE-ToS benchmark dataset (`coastalcph/lex_glue` `unfair_tos` subset / `LawInformedAI/claudette_tos`).
- **License**: Creative Commons Attribution 4.0 International (**CC BY 4.0**).
- **Split Structure (Partitioned strictly by Document)**:
  - **Train**: 5,532 clauses (4,902 Fair, 630 Unfair; 11.39% positive).
  - **Validation**: 2,275 clauses (2,045 Fair, 230 Unfair; 10.11% positive).
  - **Test**: 1,607 clauses (1,435 Fair, 172 Unfair; 10.70% positive).
- **Class Imbalance**: ~8.12 to 1 (Negative to Positive ratio).

---

## 4. Empirical Benchmark Performance (Measured on Held-out Test Set)

All metrics reflect actual, non-fabricated runs on the unseen Test Set (1,607 clauses):

| Model | Unfair Precision | Unfair Recall | Unfair F1 | PR-AUC | False Negatives (Missed) | False Positives (Alarms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Keyword Rule Baseline** | 24.4% | 38.4% | 0.2986 | 0.1598 | 106 | 204 |
| **TF-IDF + Logistic Regression** | 65.2% | 87.2% | 0.7463 | 0.8545 | 22 | 80 |
| **PyTorch ANN (Default thr = 0.50)** | 64.1% | 73.8% | 0.6865 | 0.7866 | 45 | 71 |
| **PyTorch ANN (Tuned thr = 0.024)** | 44.1% | **85.5%** | 0.5822 | 0.7866 | **25** | 186 |

---

## 5. Critical Risks & Ethical Considerations

### Mandatory Operational Warnings:
1. **Domain Mismatch (General Consumer vs Indian Finance Terms)**:
   - The model was trained on general consumer digital contracts (social networks, cloud platforms, online media), **not specifically on Indian banking, NBFC, SEBI broker, or RBI-regulated fintech documents**.
   - Performance on jurisdiction-specific finance terms (e.g. Indian IT Act mandates, SEBI power of attorney, RBI digital lending guidelines) is **untested** unless domain-labeled finance examples are provided.
2. **False Negatives are the Most Dangerous Failure**:
   - A False Negative means a predatory, one-sided clause was **not flagged**. Users must be explicitly advised that the absence of a flag does not imply a clause is fair.
3. **Temporal Drift & Changing Regulations**:
   - Contractual terms, statutes, and regulatory guidelines evolve constantly. The statistical model reflects historical patterns and possesses no real-time knowledge of newly enacted laws.
4. **No LLM Hallucinations**:
   - All explanations, tags, and notes are generated strictly from deterministic templates and human-verified knowledge-base records.

---

## 6. What We Learned (Engineering Reflections)

1. **Class Imbalance Dominates Everything**:
   - With ~11% positive prevalence, standard Cross-Entropy loss collapses toward majority-class prediction. Incorporating `pos_weight = 7.78` into `BCEWithLogitsLoss` and tuning thresholds against PR-AUC curves was essential for achieving >85% recall.
2. **Simplicity vs Complexity (Logistic Regression vs ANN)**:
   - On sparse bag-of-words features with ~5,500 training points, Logistic Regression is exceptionally tough to beat (PR-AUC 0.8545 vs ANN 0.7866). Neural networks introduce non-linear parameters that risk minor overfitting on sparse TF-IDF text without pre-trained dense embeddings.
3. **Guardrails & Abstention Over Model Confidence**:
   - An uncertainty band (marking predictions close to the decision boundary for manual review) and an abstention fallback (*"No verified reference found"*) prevent misleading or false assurances.

---

## 7. 5-Slide Presentation Outline

- **Slide 1: Problem & Motivation**
  - Why Terms & Conditions in fintech/investing are dangerous for consumers.
  - The goal: a transparent, lightweight neural network that highlights unfair clauses without legal hallucinations.
- **Slide 2: Data Pipeline & Class Imbalance**
  - CLAUDETTE-ToS corpus: 9,414 clauses partitioned by document.
  - The reality of an 8:1 class imbalance and why overall accuracy is a vanity metric.
- **Slide 3: Architecture & Training (PyTorch ANN)**
  - 5,000 TF-IDF features -> Linear(256) -> ReLU -> Dropout(0.3) -> Linear(64) -> ReLU -> Linear(1).
  - Pos-weighted BCE loss and PR-AUC monitoring with early stopping.
- **Slide 4: Benchmark Results & Honest Evaluation**
  - Side-by-side comparison: Keyword Rule vs Logistic Regression vs PyTorch ANN.
  - Threshold tuning prioritizing recall (85.5% test recall).
  - Transparent error analysis: false positives vs subtle false negatives.
- **Slide 5: Ethical Guardrails, Knowledge Base & Next Steps**
  - Verified regulatory retrieval, uncertainty bands, and strict disclaimers.
  - Next steps: Hand-labeling 200 Indian finance clauses and evaluating transfer performance.
