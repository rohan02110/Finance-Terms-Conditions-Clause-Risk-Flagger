"""
Vercel Serverless Function: Clause Risk Flagger API.

Handles POST requests with JSON payload:
{
    "text": "Terms of service text...",
    "threshold": 0.05,
    "uncertainty_band": 0.04
}

Runs:
1. Rule-based clause segmentation.
2. Lightweight TF-IDF vectorization.
3. Feed-Forward Neural Network forward pass.
4. Rule-based risk category tagging.
5. Knowledge base retrieval & grounded explanation generation.
"""

from http.server import BaseHTTPRequestHandler
import json
import os
import re
import math
from typing import List, Dict, Any

# Cached model data
MODEL_DATA = None

def load_model_data():
    global MODEL_DATA
    if MODEL_DATA is not None:
        return MODEL_DATA

    # Try local paths
    paths = [
        os.path.join(os.path.dirname(__file__), "..", "public", "model_data.json"),
        os.path.join(os.path.dirname(__file__), "model_data.json"),
        "public/model_data.json"
    ]
    for p in paths:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                MODEL_DATA = json.load(f)
                return MODEL_DATA

    raise FileNotFoundError("model_data.json not found.")


def segment_text(text: str) -> List[str]:
    if not text or not text.strip():
        return []
    cleaned = text.replace('\r\n', '\n').replace('\r', '\n').replace('\u00a0', ' ')
    pattern = re.compile(
        r'(?<=[.!?])\s+(?=[A-Z0-9\(\[\"\'])|'
        r'\n\s*\n+|'
        r'(?<=\n)\s*(?=[0-9]+[\.\)]\s+|[a-z][\.\)]\s+|\([0-9a-z]+\)\s+|[-•*]\s+)'
    )
    raw = pattern.split(cleaned)
    clauses = []
    for c in raw:
        clean_c = re.sub(r'\s+', ' ', c).strip()
        if len(clean_c) >= 10:
            clauses.append(clean_c)
    return clauses


CATEGORY_PATTERNS = {
    "unilateral_change": [
        r"\bmodif(y|ied|ication|ications)\s+(these|the|our)?\s*terms\b",
        r"\bchange\s+(these|the|our)?\s*(terms|prices|fees|agreement)\b",
        r"\bat\s+(any\s+time|our\s+sole\s+discretion)\s+without\s+(prior\s+)?notice\b",
        r"\breserves?\s+the\s+right\s+to\s+(change|amend|alter|modify)\b",
        r"\bwithout\s+prior\s+notice\b"
    ],
    "limitation_of_liability": [
        r"\b(limitation|limit)\s+of\s+liability\b",
        r"\bas[\s-]is\b",
        r"\bas[\s-]available\b",
        r"\bdisclaim\w*\s+all\s+warranties\b",
        r"\bnot\s+liable\s+for\s+(any\s+)?(direct|indirect|consequential|incidental|punitive)\b",
        r"\bhold\s+harmless\b",
        r"\bindemnif\w*\b",
        r"\bmaximum\s+aggregate\s+liability\b",
        r"\bat\s+your\s+own\s+risk\b"
    ],
    "arbitration": [
        r"\b(mandatory|binding)\s+arbitration\b",
        r"\barbitrat\w*\s+on\s+an\s+individual\s+basis\b",
        r"\bwaive\w*\s+(the\s+)?right\s+to\s+(a\s+jury\s+trial|class\s+action)\b",
        r"\bclass\s+action\s+waiver\b",
        r"\bno\s+class\s+actions\b"
    ],
    "unilateral_termination": [
        r"\bterminate\s+your\s+(access|account|membership)\b",
        r"\bsuspend\s+(or\s+terminate\s+)?your\s+account\b",
        r"\bwithout\s+cause\s+or\s+notice\b",
        r"\bin\s+(our|its)\s+sole\s+discretion\b",
        r"\bcease\s+providing\s+(the\s+)?service\b"
    ],
    "account_freeze_liquidation": [
        r"\bfreeze\s+(your\s+)?(account|funds|assets|credit)\b",
        r"\bliquidat\w*\s+(your\s+)?(holdings|positions|assets|collateral)\b",
        r"\bmargin\s+call\b",
        r"\bset[\s-]off\b",
        r"\bwithhold\s+(payment|funds)\b"
    ],
    "auto_renewal": [
        r"\bautomatic(ally)?\s+renew\w*\b",
        r"\brecurring\s+(charge|fee|payment|billing)\b",
        r"\bnon-refundable\b",
        r"\bcontinuous\s+subscription\b",
        r"\brenews\s+automatically\b"
    ],
    "data_sharing": [
        r"\bshare\s+your\s+(personal|financial|transaction)\s+data\b",
        r"\bthird[\s-]party\s+(partners|processors|affiliates)\b",
        r"\bcredit\s+(reporting\s+agency|bureau)\b",
        r"\btransfer\s+(your\s+)?data\s+outside\b"
    ]
}

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


def tag_clause(text: str) -> List[str]:
    matched = []
    for cat, patterns in CATEGORY_PATTERNS.items():
        for p in patterns:
            if re.search(p, text, re.IGNORECASE):
                matched.append(cat)
                break
    return matched if matched else ["general_risk"]


def compute_tf_idf_vector(text: str, vocab: Dict[str, int], idf: List[float]) -> List[float]:
    token_pattern = re.compile(r'(?u)\b\w\w+\b')
    words = [w.lower() for w in token_pattern.findall(text)]
    ngrams = list(words)
    for i in range(len(words) - 1):
        ngrams.append(words[i] + ' ' + words[i+1])

    counts = {}
    for ng in ngrams:
        if ng in vocab:
            counts[ng] = counts.get(ng, 0) + 1

    vec = [0.0] * len(vocab)
    sum_sq = 0.0
    for ng, count in counts.items():
        idx = vocab[ng]
        tf = 1.0 + math.log(count)
        val = tf * idf[idx]
        vec[idx] = val
        sum_sq += val * val

    if sum_sq > 0:
        norm = math.sqrt(sum_sq)
        vec = [v / norm for v in vec]
    return vec


def predict_probability(x: List[float], weights: Dict[str, Any]) -> float:
    fc1_w = weights["fc1_w"]  # (256, 5000)
    fc1_b = weights["fc1_b"]  # (256,)
    fc2_w = weights["fc2_w"]  # (64, 256)
    fc2_b = weights["fc2_b"]  # (64,)
    fc_out_w = weights["fc_out_w"][0]  # (64,)
    fc_out_b = weights["fc_out_b"][0]  # scalar

    # Layer 1
    h1 = []
    for i in range(256):
        dot = sum(w * val for w, val in zip(fc1_w[i], x) if val != 0.0) + fc1_b[i]
        h1.append(max(0.0, dot))  # ReLU

    # Layer 2
    h2 = []
    for j in range(64):
        dot = sum(w * val for w, val in zip(fc2_w[j], h1) if val != 0.0) + fc2_b[j]
        h2.append(max(0.0, dot))  # ReLU

    # Output logit
    logit = sum(w * val for w, val in zip(fc_out_w, h2) if val != 0.0) + fc_out_b
    prob = 1.0 / (1.0 + math.exp(-max(min(logit, 30.0), -30.0)))
    return prob


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)

        try:
            req = json.loads(post_data.decode("utf-8"))
        except Exception:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Invalid JSON"}).encode("utf-8"))
            return

        text = req.get("text", "")
        threshold = float(req.get("threshold", 0.05))
        uncertainty_band = float(req.get("uncertainty_band", 0.04))

        if not text.strip():
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"clauses": [], "summary": {"total": 0, "flagged": 0, "uncertain": 0}}).encode("utf-8"))
            return

        model_data = load_model_data()
        vocab = model_data["vocab"]
        idf = model_data["idf"]
        weights = model_data["weights"]
        kb_entries = model_data.get("knowledge_base", [])

        clauses = segment_text(text)
        results = []

        for c in clauses:
            x_vec = compute_tf_idf_vector(c, vocab, idf)
            prob = predict_probability(x_vec, weights)

            dist = abs(prob - threshold)
            is_uncertain = dist <= uncertainty_band

            if prob >= threshold:
                is_flagged = True
                status = "Uncertain: Review Manually" if is_uncertain else "Flagged: Potentially Risky"
            else:
                is_flagged = is_uncertain
                status = "Uncertain: Review Manually" if is_uncertain else "Standard / Not Flagged"

            categories = tag_clause(c) if is_flagged else []
            primary_cat = categories[0] if categories else "general_risk"

            cat_desc = CATEGORY_PLAIN_EXPLANATIONS.get(primary_cat, CATEGORY_PLAIN_EXPLANATIONS["general_risk"])
            explanation = f"This clause may matter because {cat_desc}"

            results.append({
                "clause_text": c,
                "predicted_probability": round(prob, 4),
                "is_flagged": is_flagged,
                "is_uncertain": is_uncertain,
                "status_label": status,
                "categories": categories,
                "primary_category": primary_cat,
                "explanation": explanation,
                "reference_note": None
            })

        flagged_count = sum(1 for r in results if r["is_flagged"])
        uncertain_count = sum(1 for r in results if r["is_uncertain"])

        response_payload = {
            "clauses": results,
            "summary": {
                "total": len(results),
                "flagged": flagged_count,
                "uncertain": uncertain_count
            }
        }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(response_payload).encode("utf-8"))
