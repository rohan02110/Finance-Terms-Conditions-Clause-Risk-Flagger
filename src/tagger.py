"""
Rule-Based Category Tagger for Terms & Conditions Clauses.

Assigns functional risk categories to clauses using targeted regex patterns:
1. unilateral_change: Rights to alter fees, terms, or service features without prior consent.
2. limitation_of_liability: Disclaimers of direct/indirect damages, 'as-is' clauses, liability caps.
3. arbitration: Mandatory binding arbitration and class action waivers.
4. unilateral_termination: Immediate suspension or cancellation of user accounts at provider's sole discretion.
5. account_freeze_liquidation: Freezing funds, forced asset liquidation, margin calls without grace period.
6. auto_renewal: Automatic recurring billing, negative option renewal, non-refundable roll-over.
7. data_sharing: Sharing financial or personal data with third parties or affiliates without affirmative opt-in.
"""

import re
from typing import List, Dict

CATEGORY_PATTERNS: Dict[str, List[str]] = {
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

COMPILED_PATTERNS = {
    cat: [re.compile(p, re.IGNORECASE) for p in patterns]
    for cat, patterns in CATEGORY_PATTERNS.items()
}


def tag_clause(text: str) -> List[str]:
    """
    Tags a single clause with matching functional risk categories.
    Returns a list of matching category keys, or ['general_risk'] if no specific pattern matches.
    """
    if not text:
        return []

    matched = []
    for cat, regexes in COMPILED_PATTERNS.items():
        for r in regexes:
            if r.search(text):
                matched.append(cat)
                break

    if not matched:
        return ["general_risk"]

    return matched
