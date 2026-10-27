# Finance T&C Clause Risk Flagger

> **Educational Decision-Support Tool**: Flags potentially unfair, one-sided, or predatory clauses in finance and investment Terms & Conditions using a PyTorch Feed-Forward Neural Network (ANN/MLP) and verified reference retrieval.
>
> ⚠️ **MANDATORY DISCLAIMER**: This tool highlights clauses for your attention. It **never** gives legal or financial advice and **never** says a document is "safe" or "low risk".

---

## 1. Project Overview

Digital investment and fintech services often feature lengthy Terms & Conditions containing clauses that heavily favor the service provider—such as unilateral modification rights, sweeping limitation of liability, mandatory binding arbitration, auto-renewal trapdoors, or immediate liquidation permissions.

This repository provides an end-to-end, reproducible, CPU-friendly system:
1. **Clause Segmentation**: Rule-based sentence and clause parsing from raw text or PDF documents (`src/segment.py`).
2. **ANN Risk Classifier**: A 2-hidden-layer feed-forward PyTorch neural network tuned specifically for **high recall** on the minority unfair class (`src/model.py`, `src/train.py`).
3. **Category Tagging**: Deterministic keyword/regex patterns mapping flagged clauses to functional risk categories (`src/tagger.py`).
4. **Knowledge Retrieval**: Lightweight TF-IDF similarity matching against a human-verified regulatory knowledge base (`src/retrieve.py`).
5. **Template Explanations**: Transparent, non-hallucinated explanations with manual review warnings for uncertain predictions (`src/explain.py`).
6. **Streamlit App**: Clean local web interface with a persistent non-dismissible disclaimer (`app/streamlit_app.py`).

---

## 2. Repository Structure

```
.
├── README.md
├── requirements.txt
├── hello_world.py
├── data/
│   ├── raw/                 # Downloaded CLAUDETTE datasets (never edited manually)
│   ├── processed/           # Document-partitioned train, val, test splits
│   └── knowledge_base/      # rules_template.csv (for your curated regulations)
├── notebooks/
│   ├── 01_explore_data.ipynb# Dataset EDA, lengths, class balance, 10 samples/class
│   └── 02_baselines.ipynb   # Keyword rule & Logistic Regression inspection
├── src/
│   ├── segment.py           # Text -> Clauses parser
│   ├── features.py          # TF-IDF feature extractor (train-fitted only)
│   ├── model.py             # PyTorch MLP Neural Network
│   ├── train.py             # Multi-seed training loop with pos_weight BCE loss
│   ├── evaluate.py          # Held-out Test set evaluation & PR-AUC analysis
│   ├── tagger.py            # Rule-based category tagger
│   ├── retrieve.py          # TF-IDF retrieval over knowledge base
│   └── explain.py           # Deterministic template builder + abstention
├── app/
│   └── streamlit_app.py     # Streamlit interface with PDF extraction
├── tests/
│   ├── test_smoke.py        # Environment & import verification
│   ├── test_segment.py      # Clause segmentation tests
│   ├── test_tagger.py       # Rule-based tagger tests
│   ├── test_retrieve.py     # Knowledge base retrieval & fallback tests
│   └── test_explain.py      # Explanation & uncertainty band tests
└── reports/
    ├── data_summary.md      # Data summary & class imbalance report
    ├── baselines_summary.md # Baseline validation benchmark
    ├── evaluation.md        # Real test set benchmarks & error analysis
    ├── model_card.md        # Comprehensive Model Card & Presentation Outline
    └── figures/             # Loss curves and PR trade-off plots
```

---

## 3. Setup & Quickstart

### Prerequisites
- Python 3.11+ (tested on Python 3.14 64-bit on Windows)
- CPU-only execution (runs locally on any laptop in minutes)

### Installation
```bash
# 1. Clone or navigate to the project directory
cd "Project Codebase"

# 2. Install dependencies
pip install -r requirements.txt
```

### Run Tests
```bash
python -m pytest tests/ -v
```

---

## 4. How to Reproduce All Phases

### Step 1: Run Exploratory Data Analysis & Baselines
```bash
# Generate EDA and Baseline summary metrics
python scripts/run_baselines.py
```

### Step 2: Train the PyTorch ANN Model
```bash
# Trains across 3 random seeds (42, 123, 999) with early stopping
python -m src.train
```

### Step 3: Run Final Held-out Test Set Evaluation
```bash
# Evaluates Keyword, Logistic Regression, and ANN on unseen test contracts
python -m src.evaluate
```

### Step 4: Run the Vercel Web App Locally
```bash
# Start local preview server on port 3000
python -m http.server 3000 --directory public
```
Open **`http://localhost:3000`** in your browser.

### Step 5: Deploy to Vercel (1-Click / CLI)
This project is pre-configured with [vercel.json](file:///d:/COLLEGE/SEM%203/Neural%20network/Project%20Codebase/vercel.json) and a lightweight Serverless API (`api/analyze.py`):
```bash
# Deploy instantly using Vercel CLI
npx vercel
```
Or push this repository to GitHub and import it directly into your [Vercel Dashboard](https://vercel.com).
- **Frontend**: High-speed vanilla HTML5/CSS3/JS with rich dark-mode fintech aesthetics, glassmorphism, and responsive layout.
- **Serverless API**: `api/analyze.py` routes requests through `/api/analyze` with zero bulky dependencies.
- **Client-Side Fallback**: `public/app.js` can also run the neural network forward pass client-side with 0ms cold starts.

---

## 5. Summary of Empirical Test Results

Evaluated on 1,607 held-out test clauses (172 positive / unfair):

| Model | Unfair Recall | Unfair Precision | F1-Score | PR-AUC |
| :--- | :---: | :---: | :---: | :---: |
| **Keyword Rule Baseline** | 38.4% | 24.4% | 0.2986 | 0.1598 |
| **TF-IDF + Logistic Regression** | 87.2% | 65.2% | **0.7463** | **0.8545** |
| **PyTorch ANN (Default thr = 0.50)** | 73.8% | 64.1% | 0.6865 | 0.7866 |
| **PyTorch ANN (Recall-Tuned thr = 0.024)** | **85.5%** | 44.1% | 0.5822 | 0.7866 |

Detailed discussion and 30 concrete error analysis cases can be viewed in [reports/evaluation.md](file:///d:/COLLEGE/SEM%203/Neural%20network/Project%20Codebase/reports/evaluation.md).

---

## 6. Curating Your Regulatory Knowledge Base

To add your own verified regulatory notes:
1. Open [data/knowledge_base/rules_template.csv](file:///d:/COLLEGE/SEM%203/Neural%20network/Project%20Codebase/data/knowledge_base/rules_template.csv).
2. Save a copy as `data/knowledge_base/rules.csv`.
3. Add rows with columns:
   `id, category, title, summary, source_name, source_url, date_verified, verified_by`
4. The system and Streamlit app automatically pick up entries from `rules.csv`. If no verified rule matches a clause, the system safely abstains with *"No verified reference found"*.
