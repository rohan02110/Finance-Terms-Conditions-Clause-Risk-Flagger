"""
Unit tests for src/tagger.py.
"""
from src.tagger import tag_clause


def test_tag_unilateral_change():
    text = "We reserve the right to change these terms at any time without prior notice."
    cats = tag_clause(text)
    assert "unilateral_change" in cats


def test_tag_limitation_of_liability():
    text = "The services are provided on an as-is basis and we disclaim all warranties."
    cats = tag_clause(text)
    assert "limitation_of_liability" in cats


def test_tag_arbitration():
    text = "You agree to resolve disputes solely through binding arbitration and waive class action rights."
    cats = tag_clause(text)
    assert "arbitration" in cats


def test_tag_account_freeze():
    text = "The broker may freeze your account and liquidate your holdings upon margin call."
    cats = tag_clause(text)
    assert "account_freeze_liquidation" in cats


def test_tag_auto_renewal():
    text = "Your subscription renews automatically with recurring charges until cancelled."
    cats = tag_clause(text)
    assert "auto_renewal" in cats


def test_tag_fallback():
    text = "Thank you for using our digital application."
    cats = tag_clause(text)
    assert cats == ["general_risk"]
