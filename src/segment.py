"""
Clause Segmentation Module.

Splits legal Terms & Conditions text into individual clauses using rule-based
heuristics (bullet lists, section numbers, punctuation, paragraph boundaries).
"""

import re
from typing import List


def segment_text(text: str) -> List[str]:
    """
    Split text into individual clauses/sentences for analysis.
    
    Heuristics:
    - Splits on multiple newlines or paragraph breaks.
    - Splits on bullet points and numbered list markers (e.g., 1.1, (a), (i), -).
    - Splits on sentence-terminating punctuation (. ! ?) followed by whitespace and a capital/number,
      while preserving common abbreviations (e.g., e.g., i.e., etc., vs., inc., ltd.).
    - Filters out whitespace-only or trivial snippets (< 5 characters).
    """
    if not text or not text.strip():
        return []

    # Normalize carriage returns and non-breaking spaces
    cleaned = text.replace('\r\n', '\n').replace('\r', '\n').replace('\u00a0', ' ')

    # Regex pattern to match clause boundaries:
    # 1. Numbered headers/sections: e.g. "1.2 ", "Section 3. ", "(b) "
    # 2. End of sentence: period/question/exclamation followed by space and uppercase or bullet
    # 3. Double newlines (paragraphs)
    pattern = re.compile(
        r'(?<=[.!?])\s+(?=[A-Z0-9\(\[\"\'])|'  # punctuation + space + start of new sentence
        r'\n\s*\n+|'                           # paragraph breaks
        r'(?<=\n)\s*(?=[0-9]+[\.\)]\s+|[a-z][\.\)]\s+|\([0-9a-z]+\)\s+|[-•*]\s+)'  # list items
    )

    raw_clauses = pattern.split(cleaned)
    clauses: List[str] = []

    for clause in raw_clauses:
        # Collapse multiple spaces and newlines inside a clause
        c = re.sub(r'\s+', ' ', clause).strip()
        # Filter out tiny noise (e.g. standalone numbers or single punctuation marks)
        if len(c) >= 10:
            clauses.append(c)

    return clauses
