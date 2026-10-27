"""
Explanation and Abstention Module.

Builds structured, transparent explanations from deterministic templates.
Zero LLM or hallucinated text: all output is grounded in the clause text,
model output probabilities, category tagger, and human-verified knowledge base.
"""

from typing import Dict, Any, List, Optional
from src.tagger import tag_clause
from src.retrieve import KnowledgeBaseRetriever

# Standard category impact descriptions
CATEGORY_PLAIN_EXPLANATIONS = {
    "unilateral_change": "it allows the service provider to modify fees, features, or contract obligations at any time without requiring your prior consent.",
    "limitation_of_liability": "it restricts or disclaims provider liability, eliminates warranties, or requires you to accept services 'as-is' with limited legal recourse.",
    "arbitration": "it waives your right to pursue disputes in court or participate in consumer class action lawsuits, requiring binding private arbitration.",
    "unilateral_termination": "it gives the company sole discretion to terminate or suspend your account and access without prior warning or cause.",
    "account_freeze_liquidation": "it permits the platform to freeze your funds or forcibly liquidate your assets/positions without an adequate grace period.",
    "auto_renewal": "it enrolls you in automatic recurring billing or non-refundable renewal charges unless cancelled within narrow deadlines.",
    "data_sharing": "it allows sharing or transferring your personal or financial records to third parties and affiliated companies.",
    "general_risk": "its language and contractual structure statistically match terms known to create significant imbalance against the consumer."
}


class ClauseExplainer:
    """
    Constructs deterministic risk explanations and applies calibrated
    abstention bands for borderline predictions.
    """

    def __init__(
        self,
        retriever: Optional[KnowledgeBaseRetriever] = None,
        threshold: float = 0.05,
        uncertainty_band: float = 0.05
    ):
        self.retriever = retriever or KnowledgeBaseRetriever()
        self.threshold = threshold
        self.uncertainty_band = uncertainty_band

    def explain(self, clause_text: str, predicted_probability: float) -> Dict[str, Any]:
        """
        Builds a verified explanation card for a given clause and model score.
        """
        # Determine risk classification and uncertainty
        dist_from_threshold = abs(predicted_probability - self.threshold)
        is_uncertain = dist_from_threshold <= self.uncertainty_band

        if predicted_probability >= self.threshold:
            is_flagged = True
            status_label = "Uncertain: Review Manually" if is_uncertain else "Flagged: Potentially Risky"
        else:
            is_flagged = is_uncertain  # Flag borderline below-threshold clauses for user attention
            status_label = "Uncertain: Review Manually" if is_uncertain else "Standard / Not Flagged"

        # Determine category tags
        categories = tag_clause(clause_text) if is_flagged else []
        primary_cat = categories[0] if categories else "general_risk"

        # Query verified knowledge base
        ref_notes = self.retriever.retrieve(clause_text, top_k=1)
        top_ref = ref_notes[0] if ref_notes else None

        # Build grounded explanation sentence
        if top_ref and top_ref.get("id") is not None:
            explanation_sentence = f"This clause may matter because: {top_ref['summary']} (Ref: {top_ref['title']} - {top_ref['source_name']})."
        else:
            cat_desc = CATEGORY_PLAIN_EXPLANATIONS.get(primary_cat, CATEGORY_PLAIN_EXPLANATIONS["general_risk"])
            explanation_sentence = f"This clause may matter because {cat_desc}"

        return {
            "clause_text": clause_text,
            "predicted_probability": round(float(predicted_probability), 4),
            "is_flagged": is_flagged,
            "is_uncertain": is_uncertain,
            "status_label": status_label,
            "categories": categories,
            "primary_category": primary_cat,
            "explanation": explanation_sentence,
            "reference_note": top_ref
        }
