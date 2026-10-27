"""
Unit tests for src/explain.py.
"""
from src.explain import ClauseExplainer


def test_explain_flagged_clause():
    explainer = ClauseExplainer(threshold=0.20, uncertainty_band=0.05)
    clause = "We may modify these terms at any time in our sole discretion."
    result = explainer.explain(clause, predicted_probability=0.85)

    assert result["is_flagged"] is True
    assert result["is_uncertain"] is False
    assert result["status_label"] == "Flagged: Potentially Risky"
    assert "unilateral_change" in result["categories"]
    assert "This clause may matter because" in result["explanation"]


def test_explain_uncertain_clause():
    explainer = ClauseExplainer(threshold=0.20, uncertainty_band=0.05)
    clause = "The platform reserves rights regarding system updates."
    # Probability within 0.20 +/- 0.05
    result = explainer.explain(clause, predicted_probability=0.22)

    assert result["is_flagged"] is True
    assert result["is_uncertain"] is True
    assert "Uncertain: Review Manually" in result["status_label"]


def test_explain_standard_clause():
    explainer = ClauseExplainer(threshold=0.20, uncertainty_band=0.05)
    clause = "Thank you for creating an account with our platform."
    result = explainer.explain(clause, predicted_probability=0.02)

    assert result["is_flagged"] is False
    assert result["is_uncertain"] is False
    assert result["status_label"] == "Standard / Not Flagged"
