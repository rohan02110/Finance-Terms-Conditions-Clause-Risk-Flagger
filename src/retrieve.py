"""
Knowledge Base Retrieval Module.

Matches flagged clauses against user-verified regulatory notes using
TF-IDF cosine similarity. Returns top-2 matches or abstains if similarity
is below the confidence threshold.
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class KnowledgeBaseRetriever:
    """
    Lightweight TF-IDF retriever over human-curated regulatory knowledge base.
    """

    def __init__(self, kb_path: str = "data/knowledge_base/rules.csv", min_similarity: float = 0.15):
        self.kb_path = kb_path
        self.min_similarity = min_similarity
        self.kb_df: pd.DataFrame = pd.DataFrame()
        self.vectorizer: TfidfVectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
        self.kb_vectors = None
        self.load_knowledge_base()

    def load_knowledge_base(self):
        """Loads and indexes the knowledge base if available."""
        if not os.path.exists(self.kb_path) or os.path.getsize(self.kb_path) == 0:
            self.kb_df = pd.DataFrame()
            return

        try:
            df = pd.read_csv(self.kb_path)
            # Check required columns
            required_cols = {"id", "category", "title", "summary", "source_name", "source_url"}
            if not required_cols.issubset(set(df.columns)) or len(df) == 0:
                self.kb_df = pd.DataFrame()
                return

            self.kb_df = df.dropna(subset=["title", "summary"]).reset_index(drop=True)
            if len(self.kb_df) == 0:
                return

            # Combine title and summary for indexing
            corpus = (self.kb_df["title"] + " " + self.kb_df["summary"]).tolist()
            self.kb_vectors = self.vectorizer.fit_transform(corpus)
        except Exception as e:
            print(f"Warning: Failed to load knowledge base from {self.kb_path}: {e}")
            self.kb_df = pd.DataFrame()

    def retrieve(self, clause_text: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """
        Retrieves up to top_k matching regulatory notes for a clause.
        If no matches exceed min_similarity or KB is empty, returns an abstention record.
        """
        if not clause_text or self.kb_df.empty or self.kb_vectors is None:
            return [{
                "id": None,
                "category": None,
                "title": "No verified reference found",
                "summary": "No verified regulatory reference is available in the current knowledge base.",
                "source_name": None,
                "source_url": None,
                "similarity": 0.0
            }]

        clause_vec = self.vectorizer.transform([clause_text])
        similarities = cosine_similarity(clause_vec, self.kb_vectors)[0]

        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []

        for idx in top_indices:
            score = float(similarities[idx])
            if score >= self.min_similarity:
                row = self.kb_df.iloc[idx]
                results.append({
                    "id": row.get("id"),
                    "category": row.get("category"),
                    "title": row.get("title"),
                    "summary": row.get("summary"),
                    "source_name": row.get("source_name"),
                    "source_url": row.get("source_url"),
                    "similarity": round(score, 3)
                })

        if not results:
            return [{
                "id": None,
                "category": None,
                "title": "No verified reference found",
                "summary": "The clause text did not exceed the relevance similarity threshold against verified references.",
                "source_name": None,
                "source_url": None,
                "similarity": 0.0
            }]

        return results
