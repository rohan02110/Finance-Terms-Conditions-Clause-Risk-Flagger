# Data Summary Report: CLAUDETTE-ToS Dataset

## 1. Dataset Provenance & Identification

- **Dataset Identifier**: `coastalcph/lex_glue` (`unfair_tos` subset) / `LawInformedAI/claudette_tos`
- **Origin**: CLAUDETTE Project (Lippi et al., 2019; Chalkidis et al., 2022)
- **License**: Creative Commons Attribution 4.0 International (**CC BY 4.0**)
- **Data Source**: 50 Terms of Service contracts from major online consumer platforms (e.g., social media, content, gaming, communication).
- **Partitioning Strategy**: **Document-level partitioning** (LexGLUE benchmark split) to prevent data leakage of clauses from the same contract across splits.

---

## 2. Split Sizes and Class Imbalance

| Split | Total Clauses | Fair / Standard (Class 0) | Potentially Unfair / Risky (Class 1) | % Unfair | Imbalance Ratio (Neg : Pos) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | 5,532 | 4,902 | 630 | 11.39% | 7.78 : 1 |
| **Validation** | 2,275 | 2,045 | 230 | 10.11% | 8.89 : 1 |
| **Test** | 1,607 | 1,435 | 172 | 10.70% | 8.34 : 1 |
| **Full Corpus** | **9,414** | **8,382** | **1,032** | **10.96%** | **8.12 : 1** |

> **Critical Takeaway on Class Imbalance**:
> Across all splits, only ~10.7% to 11.4% of clauses belong to the positive (unfair/risky) class. 
> A trivial majority-class classifier that predicts "Fair" for every single clause would achieve **89.04% accuracy**, yet catch **0%** of predatory clauses. 
> Therefore, overall accuracy is a misleading metric; evaluation must focus on **Precision, Recall, F1-score on the risky class**, and **Precision-Recall AUC (PR-AUC)**.

---

## 3. Clause Length & Word Count Distributions (Train Set)

| Class | Count | Mean Words | Std Dev | Min Words | Median (50%) | Max Words |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fair (0)** | 4,902 | 31.8 | 21.6 | 6 | 26.0 | 441 |
| **Unfair / Risky (1)** | 630 | 36.6 | 21.7 | 6 | 31.0 | 195 |

- Unfair clauses have a slightly higher median word count (31 vs 26 words) and mean length (36.6 vs 31.8 words), often containing multi-clause compound sentences specifying disclaimers and discretionary rights.

---

## 4. Fine-Grained Category Breakdown in Unfair Clauses (Train Set)

| Category | Count in Train | % of Unfair Clauses | Core Legal Theme |
| :--- | :---: | :---: | :--- |
| **Limitation of liability** | 240 | 38.10% | Disclaimers of damages, "as-is" warranties, liability caps |
| **Unilateral change** | 134 | 21.27% | Changing terms or service features without prior consent |
| **Unilateral termination** | 99 | 15.71% | Provider right to suspend or cancel account at sole discretion |
| **Content removal** | 71 | 11.27% | Immediate removal of user content without notice |
| **Jurisdiction** | 62 | 9.84% | Forcing dispute venue to provider's home court |
| **Choice of law** | 59 | 9.37% | Foreign governing law, bypassing local consumer protection |
| **Arbitration** | 35 | 5.56% | Mandatory binding arbitration & class action waivers |
| **Contract by using** | 24 | 3.81% | Tacit agreement merely by browsing or accessing |

*(Note: Some clauses contain overlapping violations, so category counts sum to slightly more than the 630 unfair clauses).*

---

## 5. Sample Clauses (Verbatim from Corpus)

### Fair / Non-Risky Examples (Class 0):
1. *"thanks for sending us good vibes by using the viber app !"*
2. *"you can change your payment method and billing information by logging into your viber account at http://account.viber.com/ ."*
3. *"you agree that viber may use information regarding your selected payment method provided by your issuing bank or applicable payment system ."*

### Potentially Unfair / Risky Examples (Class 1):
1. *"viber may remove or modify any content without notice or liability at any time in viber 's sole discretion ."* (Unilateral change / Content removal)
2. *"any use of the forums will be at your own risk and will be subject to the disclaimers and limitations on liability set out above ."* (Limitation of liability)
3. *"we may terminate your access to and use of the services , at our sole discretion , at any time and without notice to you ."* (Unilateral termination)
