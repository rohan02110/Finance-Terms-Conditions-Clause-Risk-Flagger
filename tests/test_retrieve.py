"""
Unit tests for src/retrieve.py.
"""
import os
import tempfile
import pandas as pd
from src.retrieve import KnowledgeBaseRetriever


def test_retrieve_empty_kb():
    # When KB does not exist or has 0 rows
    retriever = KnowledgeBaseRetriever(kb_path="non_existent_kb.csv")
    results = retriever.retrieve("We may change terms without notice.")
    assert len(results) == 1
    assert results[0]["title"] == "No verified reference found"
    assert results[0]["similarity"] == 0.0


def test_retrieve_matching_and_threshold():
    # Create temporary CSV with a sample verified rule
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as tmp:
        tmp.write("id,category,title,summary,source_name,source_url,date_verified,verified_by\n")
        tmp.write("RULE-01,unilateral_change,Unilateral Modification Restrictions,Service providers must give 30 days prior notice before changing fees,Consumer Protection Council,https://example.com/law,2026-01-01,UserTester\n")
        tmp_path = tmp.name

    try:
        retriever = KnowledgeBaseRetriever(kb_path=tmp_path, min_similarity=0.15)
        
        # 1. Matching case
        match_results = retriever.retrieve("The provider will change fees and terms without giving notice.")
        assert len(match_results) >= 1
        assert match_results[0]["id"] == "RULE-01"
        assert match_results[0]["similarity"] > 0.15

        # 2. Non-matching case (irrelevant text below min_similarity)
        no_match = retriever.retrieve("The weather in London is quite rainy today.")
        assert len(no_match) == 1
        assert no_match[0]["title"] == "No verified reference found"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
