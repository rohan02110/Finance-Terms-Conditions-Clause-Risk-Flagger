"""
Features Module: TF-IDF Vectorization for Legal Clauses.

Extracts statistical unigram and bigram features from text.
Fitted strictly on training data to prevent information leakage.
"""

import os
import joblib
from typing import Tuple
from sklearn.feature_extraction.text import TfidfVectorizer


def create_vectorizer(max_features: int = 5000) -> TfidfVectorizer:
    """
    Instantiate a TF-IDF vectorizer configured for legal text:
    - unigrams + bigrams (ngram_range=(1, 2)) capture compound legal phrases
      like 'sole discretion', 'at any time', 'without notice'.
    - sublinear_tf=True applies logarithmic term frequency scaling (1 + log(tf)),
      preventing repetitive words in long boilerplate clauses from dominating.
    - min_df=2 ignores rare typos appearing only once.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        lowercase=True,
        sublinear_tf=True,
        min_df=2
    )


def fit_and_save_vectorizer(train_texts, save_path: str = "models/tfidf_vectorizer.joblib") -> TfidfVectorizer:
    """Fit vectorizer strictly on training data and persist to disk."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    vectorizer = create_vectorizer()
    vectorizer.fit(train_texts)
    joblib.dump(vectorizer, save_path)
    print(f"Fitted vectorizer saved to {save_path} with {len(vectorizer.get_feature_names_out())} features.")
    return vectorizer


def load_vectorizer(load_path: str = "models/tfidf_vectorizer.joblib") -> TfidfVectorizer:
    """Load a pre-fitted vectorizer from disk."""
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Vectorizer not found at {load_path}. Please run training first.")
    return joblib.load(load_path)
