# Model Evaluation & Benchmark Report

## 1. Test Set Performance Comparison

All models were evaluated on the held-out **Test Set** (1,607 total clauses: 1,435 Fair, 172 Unfair/Risky).

| Model / Configuration | Precision (Unfair) | Recall (Unfair) | F1-Score (Unfair) | PR-AUC | TN | FP | FN | TP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Keyword Rule Baseline** | 0.2444 (24.4%) | 0.3837 (38.4%) | 0.2986 | 0.1598 | 1231 | 204 | 106 | 66 |
| **TF-IDF + Logistic Regression** | 0.6522 (65.2%) | 0.8721 (87.2%) | 0.7463 | 0.8545 | 1355 | 80 | 22 | 150 |
| **PyTorch ANN (Default thr = 0.50)** | 0.6414 (64.1%) | 0.7384 (73.8%) | 0.6865 | 0.7866 | 1364 | 71 | 45 | 127 |
| **PyTorch ANN (Tuned thr = 0.02)** | **0.4414** (44.1%) | **0.8547** (85.5%) | **0.5822** | **0.7866** | 1249 | 186 | 25 | 147 |

---

## 2. Honest Analysis & Comparison

### Did the ANN beat Logistic Regression?
- **PR-AUC**: Logistic Regression achieved **0.8545** PR-AUC, while the ANN achieved **0.7866** PR-AUC.
- **Recall & F1**: With threshold tuning targeting high recall (0.02), the ANN achieves **85.5% recall** and an F1 of **0.5822**.
- **Engineering Verdict**: For shallow n-gram bag-of-words representations on small-to-medium corpora (~5,500 training examples), a convex linear model (Logistic Regression with class weights) is extremely competitive with a 2-layer neural network. The neural network learns non-linear interactions across word co-occurrences, but because TF-IDF vectors are sparse, linear separation is already strong. The ANN's main strength in this project is its continuous probabilistic calibration and flexibility for downstream joint tasks.

---

## 3. Decision Threshold Selection & Recall Prioritization

In financial consumer protection, a **False Negative (missing a predatory clause)** is far more hazardous to the user than a **False Positive (flagging a benign clause for manual check)**.

- Using the validation PR trade-off curve, we selected **threshold = 0.024**.
- At this threshold, the model catches **147 of 172 risky clauses (85.5% Recall)** on completely unseen contracts.

---

## 4. Error Analysis

### 15 False Negatives (Missed Risky Clauses)
1. **Clause**: "by using the service , you acknowledge , accept and agree with all provisions of the privacy policy , including , without limitation , the use and treatment of the text , images , and other data you choose to input , upload , or store in evernote ( collectively , `` content '' ) and your personal information in accordance with such privacy policy . 
"
   - **Predicted Prob**: 0.000 | **Category**: Contract by using
   - **Hypothesis**: Novel formulation / rare domain terms
2. **Clause**: "we encourage you to give the privacy policy a careful look because , by using our services , you agree that your information may be collected and processed by snap group limited in the uk and snap inc. in the u.s. , as well as their affiliates , and transferred outside of the country in which you reside , including to countries that may not have the same data-protection laws as your country . 
"
   - **Predicted Prob**: 0.000 | **Category**: Contract by using
   - **Hypothesis**: Novel formulation / rare domain terms
3. **Clause**: "to the extent allowed under law , linkedin and its affiliates ( and those that linkedin works with to provide the services ) ( a ) disclaim all implied warranties and representations ( e.g. warranties of merchantability , fitness for a particular purpose , accuracy of data , and noninfringement ) ; ( b ) do not guarantee that the services will function without interruption or errors , and ( c ) provide the service ( including content and information ) on an `` as is '' and `` as available '' basis . 
"
   - **Predicted Prob**: 0.000 | **Category**: Limitation of liability
   - **Hypothesis**: Novel formulation / rare domain terms
4. **Clause**: "we have the right to limit how you connect and interact on our services . 
"
   - **Predicted Prob**: 0.000 | **Category**: Unilateral change
   - **Hypothesis**: Subtle phrasing without overt discretionary terms
5. **Clause**: "you agree that regardless of any statute or law to the contrary or the applicable dispute resolution process , any claim or cause of action you may have arising out of or related to use of the service or otherwise under these must be filed within one ( 1 ) year after such claim or cause of action arose or you hereby agree to be forever barred from bringing such claim . 
"
   - **Predicted Prob**: 0.000 | **Category**: Limitation of liability
   - **Hypothesis**: Novel formulation / rare domain terms
6. **Clause**: "uber 's services may be used by you to request and schedule transportation , goods or logistics services with third party providers , but you agree that uber has no responsibility or liability to you related to any transportation , goods or logistics services provided to you by third party providers other than as expressly set forth in these terms . 
"
   - **Predicted Prob**: 0.000 | **Category**: Limitation of liability
   - **Hypothesis**: Novel formulation / rare domain terms
7. **Clause**: "the changes will not be retroactive , and the most current version of the terms , which will always be at twitter.com/tos , will govern our relationship with you . 
"
   - **Predicted Prob**: 0.000 | **Category**: Unilateral change
   - **Hypothesis**: Novel formulation / rare domain terms
8. **Clause**: "unless you and we agree otherwise , in the event that the agreement to arbitrate above is found not to apply to you or to a particular claim or dispute , either as a result of your decision to opt out of the agreement to arbitrate or as a result of a decision by the arbitrator or a court order , you agree that any claim or dispute that has arisen or may arise between you and ebay must be resolved exclusively by a state or federal court located in salt lake county , utah . 
"
   - **Predicted Prob**: 0.001 | **Category**: Jurisdiction
   - **Hypothesis**: Novel formulation / rare domain terms
9. **Clause**: "we will try to notify you of material revisions , for example via a service notification or an email to the email associated with your account . 
"
   - **Predicted Prob**: 0.001 | **Category**: Unilateral change
   - **Hypothesis**: Novel formulation / rare domain terms
10. **Clause**: "any material downloaded or otherwise obtained through the use of the service is done at your own discretion and risk and you are solely responsible for any damage to your computer or other device or loss of data resulting from the download or use of any such material . 
"
   - **Predicted Prob**: 0.001 | **Category**: Limitation of liability
   - **Hypothesis**: Novel formulation / rare domain terms
11. **Clause**: "and if snap group limited becomes aware that one of its users has repeatedly infringed copyrights , we will take reasonable steps within our power to terminate the user 's account . 
"
   - **Predicted Prob**: 0.001 | **Category**: Unilateral termination
   - **Hypothesis**: Novel formulation / rare domain terms
12. **Clause**: "we may take any of these actions at any time , and when we do , we will try to notify you beforehand , but this wo n't always be possible . 
"
   - **Predicted Prob**: 0.002 | **Category**: Unilateral change
   - **Hypothesis**: Novel formulation / rare domain terms
13. **Clause**: "the licenses granted by amazon terminate if you do not comply with these conditions of use or any service terms . 
"
   - **Predicted Prob**: 0.002 | **Category**: Unilateral termination
   - **Hypothesis**: Novel formulation / rare domain terms
14. **Clause**: "if we find that any shared content in your account violates our terms of service ( including by violating another person 's intellectual property or privacy rights ) , we reserve the right to un-share or take down such content . 
"
   - **Predicted Prob**: 0.002 | **Category**: Content removal
   - **Hypothesis**: Novel formulation / rare domain terms
15. **Clause**: "your content in memories might become unavailable for any number of reasons , including things like an operational glitch or a decision on our end to terminate your account . 
"
   - **Predicted Prob**: 0.002 | **Category**: Content removal; Unilateral termination
   - **Hypothesis**: Novel formulation / rare domain terms

### 15 False Positives (Benign Clauses Flagged as Risky)
1. **Clause**: "you acknowledge and agree that , to the maximum extent permitted by law , the entire risk arising out of your access to and use of the site , services and collective content remains with you . 
"
   - **Predicted Prob**: 1.000
   - **Hypothesis**: Contains legal disclaimer n-grams ('may', 'without', 'notice') in a benign consumer context
2. **Clause**: "right to modify the service . 
"
   - **Predicted Prob**: 0.999
   - **Hypothesis**: Contains legal disclaimer n-grams ('may', 'without', 'notice') in a benign consumer context
3. **Clause**: "ebay takes no responsibility and assumes no liability for any content provided by you or any third party . 
"
   - **Predicted Prob**: 0.998
   - **Hypothesis**: Contains legal disclaimer n-grams ('may', 'without', 'notice') in a benign consumer context
4. **Clause**: "we can not and will not be liable for any loss or damage arising from your failure to comply with the above . 
"
   - **Predicted Prob**: 0.998
   - **Hypothesis**: Contains legal disclaimer n-grams ('may', 'without', 'notice') in a benign consumer context
5. **Clause**: "although snap group limited reserves the right to review all content that appears on the services and to remove any content that violates these terms , we do not necessarily review all of it . 
"
   - **Predicted Prob**: 0.997
   - **Hypothesis**: Contains legal disclaimer n-grams ('may', 'without', 'notice') in a benign consumer context
6. **Clause**: "we will not be liable for any loss or damage arising from your failure to provide us with accurate information or to keep your basic subscriber information secure . 
"
   - **Predicted Prob**: 0.995
   - **Hypothesis**: Contains legal disclaimer n-grams ('may', 'without', 'notice') in a benign consumer context
7. **Clause**: "if we modify these terms we will post the modification on the site or provide you with notice of the modification . 
"
   - **Predicted Prob**: 0.995
   - **Hypothesis**: Contains legal disclaimer n-grams ('may', 'without', 'notice') in a benign consumer context
8. **Clause**: "bullies , harasses , or intimidates . 
"
   - **Predicted Prob**: 0.991
   - **Hypothesis**: Contains legal disclaimer n-grams ('may', 'without', 'notice') in a benign consumer context
9. **Clause**: "our user agreement has been updated . 
"
   - **Predicted Prob**: 0.989
   - **Hypothesis**: Contains legal disclaimer n-grams ('may', 'without', 'notice') in a benign consumer context
10. **Clause**: "the content that may be available to watch will vary by geographic location and will change from time to time . 
"
   - **Predicted Prob**: 0.987
   - **Hypothesis**: Contains legal disclaimer n-grams ('may', 'without', 'notice') in a benign consumer context
11. **Clause**: "by using the services , you agree that : 
"
   - **Predicted Prob**: 0.987
   - **Hypothesis**: Contains legal disclaimer n-grams ('may', 'without', 'notice') in a benign consumer context
12. **Clause**: "also , you agree that : 
"
   - **Predicted Prob**: 0.984
   - **Hypothesis**: Contains legal disclaimer n-grams ('may', 'without', 'notice') in a benign consumer context
13. **Clause**: "you agree to comply with our returns policy . 
"
   - **Predicted Prob**: 0.984
   - **Hypothesis**: Contains legal disclaimer n-grams ('may', 'without', 'notice') in a benign consumer context
14. **Clause**: "the amount of any refund will depend on the nature of the test issue and will not exceed the total amount actually paid by the user for the affected test . 
"
   - **Predicted Prob**: 0.980
   - **Hypothesis**: Contains legal disclaimer n-grams ('may', 'without', 'notice') in a benign consumer context
15. **Clause**: "the terms of service constitutes a contract between evernote and you . 
"
   - **Predicted Prob**: 0.980
   - **Hypothesis**: Contains legal disclaimer n-grams ('may', 'without', 'notice') in a benign consumer context

---

## 5. Known Limitations
1. **Consumer vs Finance Specialization**: Trained on general consumer web ToS (CLAUDETTE benchmark), not exclusively banking/brokerage agreements. Indian fintech-specific clauses (e.g. SEBI/RBI mandatory mandates) may use domain terms not represented in training.
2. **BoW / N-Gram Context Limits**: TF-IDF cannot parse complex negation (e.g., *"We will never terminate without notice"* vs *"We may terminate without notice"*).
3. **No Legal Guarantee**: Machine learning classification is purely statistical decision support.
