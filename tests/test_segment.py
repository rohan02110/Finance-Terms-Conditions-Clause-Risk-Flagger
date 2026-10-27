"""
Unit tests for src/segment.py.
"""
from src.segment import segment_text


def test_segment_simple():
    text = "We may change these terms at any time. You agree to arbitrate disputes."
    clauses = segment_text(text)
    assert len(clauses) == 2
    assert "change these terms" in clauses[0]
    assert "arbitrate disputes" in clauses[1]


def test_segment_bullet_points():
    text = """
    User Responsibilities:
    1. You must maintain confidentiality of your password.
    2. You agree not to reverse engineer the app.
    3. We reserve the right to terminate your account without notice.
    """
    clauses = segment_text(text)
    assert len(clauses) >= 3
    assert any("terminate your account" in c for c in clauses)


def test_segment_empty_input():
    assert segment_text("") == []
    assert segment_text("   \n\n   ") == []
