"""
Streamlit Web Application: Finance T&C Clause Risk Flagger.

An educational decision-support tool highlighting potentially unfair or
risky clauses in finance and investment Terms & Conditions.
"""

import os
import sys

# Ensure root directory is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
import torch
import pdfplumber

from src.segment import segment_text
from src.features import load_vectorizer
from src.model import ClauseRiskANN
from src.explain import ClauseExplainer
from src.retrieve import KnowledgeBaseRetriever

# Page configuration
st.set_page_config(
    page_title="T&C Clause Risk Flagger",
    page_icon="⚖️",
    layout="wide"
)

# Persistent Mandatory Disclaimer
DISCLAIMER_TEXT = (
    "⚠️ **IMPORTANT DISCLAIMER**: This tool highlights clauses for your attention and decision-making support. "
    "It **does not provide legal or financial advice** and **cannot tell you a document is safe or low-risk**. "
    "The absence of flagged clauses does not indicate that an agreement is safe to sign."
)

SAMPLE_FINANCE_TOC = """
1. Account Modification and Termination
We reserve the right to amend, alter, or modify these Terms of Service and any associated fee schedules at any time in our sole discretion without prior notice to you. 
You may terminate your account at any time by contacting customer support. 
We may suspend or terminate your account and access to the investment platform immediately without notice or cause.

2. Disclaimers and Limitation of Liability
The services and financial information are provided strictly on an 'as-is' and 'as-available' basis without warranties of any kind. 
In no event shall the platform, its affiliates, or directors be liable for any direct, indirect, consequential, or punitive damages arising from market volatility or system downtime. 
Your sole and exclusive remedy for dissatisfaction is to stop using the application.

3. Dispute Resolution and Class Action Waiver
You agree that all disputes, claims, or controversies shall be resolved exclusively through final and binding private arbitration rather than in court. 
You expressly waive any right to bring or participate in any class action lawsuit against the company.

4. Margin Calls and Asset Liquidation
In the event your account equity falls below maintenance margin requirements, the broker reserves the right to freeze your account and liquidate your holdings without prior notice. 
Recurring platform maintenance charges will be billed automatically to your designated payment method every month and are non-refundable.
"""


@st.cache_resource
def load_system_pipeline():
    """Load fitted TF-IDF vectorizer and trained PyTorch ANN model."""
    vectorizer = load_vectorizer("models/tfidf_vectorizer.joblib")
    checkpoint = torch.load("models/best_ann_model.pt", weights_only=True)
    
    model = ClauseRiskANN(input_dim=checkpoint["input_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load knowledge base retriever if present
    retriever = KnowledgeBaseRetriever(kb_path="data/knowledge_base/rules.csv", min_similarity=0.15)
    return vectorizer, model, retriever


def extract_text_from_pdf(uploaded_file) -> str:
    """Extracts raw text from an uploaded PDF file using pdfplumber."""
    text_content = []
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text_content.append(t)
    return "\n\n".join(text_content)


def main():
    # Render persistent disclaimer
    st.error(DISCLAIMER_TEXT)

    st.title("⚖️ Finance Terms & Conditions Clause Risk Flagger")
    st.markdown(
        "Analyze finance, fintech, and investment app Terms & Conditions to identify clauses that "
        "grant unilateral rights, limit provider liability, mandate binding arbitration, or permit automatic liquidation."
    )

    try:
        vectorizer, model, retriever = load_system_pipeline()
    except Exception as e:
        st.error(f"Error loading model pipeline: {e}. Please ensure Phase 3 training has been completed.")
        return

    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ Flagger Settings")
        threshold = st.slider(
            "Sensitivity Threshold",
            min_value=0.01,
            max_value=0.50,
            value=0.05,
            step=0.01,
            help="Lower values maximize recall (catching more potentially risky clauses with more review items)."
        )
        uncertainty_band = st.slider(
            "Uncertainty Band (+/-)",
            min_value=0.01,
            max_value=0.10,
            value=0.04,
            step=0.01,
            help="Clauses within this distance from the threshold are labeled 'Uncertain: Review Manually'."
        )

        st.markdown("---")
        st.markdown("### 📚 Model Information")
        st.info(
            "**Architecture**: 2-layer Feed-Forward PyTorch ANN\n\n"
            "**Features**: 5,000 TF-IDF unigram & bigram features\n\n"
            "**Objective**: Optimized for high recall on risky clauses"
        )

    # Document Input Section
    input_tab1, input_tab2, input_tab3 = st.tabs(["📝 Paste Text", "📄 Upload PDF / TXT", "💡 Load Sample Finance T&C"])

    raw_text = ""
    with input_tab1:
        raw_text_input = st.text_area("Paste terms & conditions text here:", height=220, placeholder="Paste contractual text...")
        if raw_text_input:
            raw_text = raw_text_input

    with input_tab2:
        uploaded_file = st.file_uploader("Upload a .txt or .pdf contract", type=["txt", "pdf"])
        if uploaded_file is not None:
            if uploaded_file.name.endswith(".pdf"):
                raw_text = extract_text_from_pdf(uploaded_file)
                if not raw_text.strip():
                    st.warning("⚠️ No extractable text found in this PDF. It may be an image scan without OCR.")
            else:
                raw_text = uploaded_file.read().decode("utf-8", errors="ignore")

    with input_tab3:
        if st.button("Load Sample Investment T&C Document"):
            raw_text = SAMPLE_FINANCE_TOC
            st.session_state["sample_loaded"] = True

    if not raw_text.strip():
        st.info("👆 Paste text, upload a document, or load the sample above to begin clause risk analysis.")
        return

    # Analyze Button
    if st.button("🔍 Analyze Clauses", type="primary") or st.session_state.get("sample_loaded"):
        with st.spinner("Segmenting text and scoring clauses with neural network..."):
            clauses = segment_text(raw_text)

            if not clauses:
                st.warning("No valid text clauses could be parsed. Check that the document contains full sentences.")
                return

            # Vectorize clauses
            features = vectorizer.transform(clauses).toarray()
            features_tensor = torch.tensor(features, dtype=torch.float32)

            # PyTorch inference
            probs = model.predict_proba(features_tensor).cpu().numpy().flatten()

            # Build grounded explanations
            explainer = ClauseExplainer(retriever=retriever, threshold=threshold, uncertainty_band=uncertainty_band)
            analysis_results = [explainer.explain(c, p) for c, p in zip(clauses, probs)]

        # Summary Metrics
        total_clauses = len(analysis_results)
        flagged_items = [r for r in analysis_results if r["is_flagged"]]
        uncertain_items = [r for r in analysis_results if r["is_uncertain"]]
        high_risk_items = [r for r in flagged_items if not r["is_uncertain"]]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Clauses Parsed", total_clauses)
        col2.metric("Potentially Risky Flagged", len(high_risk_items))
        col3.metric("Uncertain (Manual Review)", len(uncertain_items))
        col4.metric("Flag Rate", f"{len(flagged_items) / total_clauses * 100:.1f}%")

        st.markdown("---")

        if not flagged_items:
            st.warning(
                "ℹ️ **No clauses exceeded the risk threshold.**\n\n"
                "*Reminder: This does not mean the agreement is safe to sign.* "
                "You may lower the sensitivity threshold in the sidebar to review additional borderline clauses."
            )
            return

        # Sort flagged clauses descending by risk probability
        flagged_items.sort(key=lambda x: x["predicted_probability"], reverse=True)

        st.subheader(f"🚩 Flagged Clauses Requiring Attention ({len(flagged_items)})")

        # Category Filter
        all_categories = sorted(list({cat for item in flagged_items for cat in item["categories"]}))
        selected_cats = st.multiselect("Filter by Risk Category", all_categories, default=all_categories)

        filtered_items = [
            item for item in flagged_items
            if any(cat in selected_cats for cat in item["categories"])
        ]

        for idx, item in enumerate(filtered_items, 1):
            prob_pct = item["predicted_probability"] * 100
            status = item["status_label"]
            badge_color = "🔴" if "Potentially Risky" in status else "🟡"
            cat_badges = " ".join([f"`{c}`" for c in item["categories"]])

            header_label = f"{badge_color} [{prob_pct:.1f}% Risk Score] {cat_badges} — {item['clause_text'][:90]}..."

            with st.expander(header_label, expanded=(idx <= 3)):
                st.markdown(f"**Status**: {badge_color} **{status}** (Predicted Probability: `{item['predicted_probability']:.4f}`)")
                st.markdown(f"**Exact Clause Text**:\n> *\"{item['clause_text']}\"*")
                st.markdown(f"**Explanation**: {item['explanation']}")

                # Verified reference note
                ref = item.get("reference_note")
                if ref and ref.get("id") is not None:
                    st.success(
                        f"📖 **Verified Regulatory Reference**: [{ref['title']}]({ref['source_url']})\n\n"
                        f"{ref['summary']} *(Source: {ref['source_name']})*"
                    )
                else:
                    st.caption("ℹ️ *No verified regulatory reference note matched this clause text.*")


if __name__ == "__main__":
    main()
