# LPDG Gateway Visit Prioritisation - Decisions

## 1. Problem Definition

The goal is to help LPDG decide which gateways should receive a field visit each week.

LPDG has approximately 320 gateways, but field operations can perform a maximum of 15 site visits per week. Therefore, the system must rank gateways by operational risk and select the 15 highest-priority gateways.

The system produces predictions for eight weeks:

- 2026-02-02
- 2026-02-09
- 2026-02-16
- 2026-02-23
- 2026-03-02
- 2026-03-09
- 2026-03-16
- 2026-03-23

The final `predictions.csv` contains exactly 120 rows: 15 gateways for each of the eight weeks.

---

## 2. Definition of a Problem Gateway

The challenge does not provide a fixed definition of a gateway that "needs a visit", so an operational definition was created from the available telemetry and meter-read data.

A gateway is labelled as having a future operational problem when, during the following seven-day period, at least one of the following conditions occurs:

- Offline duration is at least 24 hours.
- Number of disconnections is at least 50.
- Number of reboots is at least 12.
- Meter-read success rate is below 80%.

This definition combines connectivity, stability, and service-quality signals rather than relying on a single telemetry metric.

The thresholds were selected to identify sustained or significant operational degradation while avoiding treating every small fluctuation as a field-service problem.

---

## 3. Prediction Window

For each gateway-week, the model uses information available before that week's prediction date.

The main feature window covers the previous 28 days.

The 28-day history is divided into:

- Recent 7 days
- Previous 21 days

This allows the model to capture both the current state and whether the gateway is deteriorating compared with its recent history.

The future seven-day period is used only to create the target during model development and evaluation.

---

## 4. Leakage Prevention

Future information must not be available to the model when making a prediction.

Therefore:

- Target variables are excluded from the model features.
- Future seven-day telemetry is used only to create evaluation labels.
- Training and validation are separated chronologically.
- The temporal validation period occurs after the training period.
- The unseen-gateway experiment uses gateways that were completely excluded from model training.

This prevents the model from learning directly from future outcomes.

---

## 5. Data Used

The model combines information from several available sources:

### Telemetry

Hourly gateway telemetry provides information about:

- Disconnections
- Offline duration
- Reboots
- Network behaviour
- Packet counts
- CRC errors
- RSSI/signal quality
- Connectivity importance measures
- Gateway availability

### Meter Read Success

Meter-read data provides:

- Expected meter reads
- Successful meter reads
- Meter-read success rate
- Minimum and average success rate over the historical window

### Historical Field Visits

Historical field visits provide operational context, including:

- Previous visit count
- Visits where an error was fixed
- Visits where no error was found
- No-access outcomes

### Gateway Master Data

Gateway metadata provides contextual information such as:

- Site type
- Region
- Hardware model
- Antenna type
- Firmware version
- Number of installed meters

These sources allow the model to combine current gateway behaviour with historical reliability and gateway characteristics.

---

## 6. Feature Engineering

Features were aggregated over the previous 28 days.

Examples include:

- Recent seven-day disconnection count
- Recent 28-day disconnection count
- Previous 21-day disconnection count
- Recent seven-day disconnection hours
- Recent 28-day disconnection hours
- Recent offline hours
- Recent coverage ratio
- No-connection importance
- Meter-read success statistics
- Historical field-visit statistics

Trend features compare recent behaviour with the previous period.

For example, increasing disconnections or increasing offline hours indicate deterioration.

This is important because a gateway that is rapidly becoming worse should receive a higher priority than one with a stable historical problem.

---

## 7. Machine Learning Model

A Random Forest classifier was selected for the ML solution.

The model configuration is:

- 400 trees
- Maximum depth: 14
- Minimum samples per leaf: 3
- Balanced class weighting
- Random state: 42

Random Forest was selected because it:

- Works well with mixed operational features.
- Can model nonlinear relationships.
- Does not require feature scaling.
- Provides feature importance information.
- Is practical to train and run on CPU.
- Can handle interactions between telemetry, network, and service-quality signals.

Categorical features are one-hot encoded and missing values are imputed through the preprocessing pipeline.

---

## 8. Ranking and Visit Selection

The model produces a probability that a gateway will experience a future operational problem.

For each prediction week:

1. All eligible gateways are scored.
2. Gateways are ranked from highest to lowest predicted risk.
3. The top 15 gateways are selected.
4. Rank 1 represents the highest predicted priority.

This approach directly respects the operational constraint of a maximum of 15 visits per week.

The model does not use a probability threshold to determine the number of visits because the number of available visits is fixed. Instead, the risk probability is used for ranking.

---

## 9. Cost Model

The challenge specifies two operational costs:

- Unnecessary field visit: €380
- Broken gateway left without a visit: €600 per week

Therefore:

**Total Cost = False Positives × €380 + False Negatives × €600**

Because a missed problem costs more than an unnecessary visit, the ranking should prioritise gateways with a high probability of future problems.

The 15-visit limit prevents simply visiting every high-risk gateway.

---

## 10. Baseline Comparison

The provided 3-Sigma baseline was used as the comparison method.

Both methods were evaluated over the seven weeks for which a complete future seven-day outcome was available:

- 2026-02-02
- 2026-02-09
- 2026-02-16
- 2026-02-23
- 2026-03-02
- 2026-03-09
- 2026-03-16

The final 2026-03-23 prediction is retained for the required eight-week submission, but it cannot be included in the cost evaluation because the available telemetry ends on 2026-03-31 and a complete future seven-day window is not available.

### Results

| Metric | 3-Sigma Baseline | ML Model |
|---|---:|---:|
| True Positives | 65 | 105 |
| False Positives | 40 | 0 |
| False Negatives | 639 | 599 |
| Precision | 61.90% | 100.00% |
| Recall | 9.23% | 14.91% |
| Total Cost | €398,600 | €359,400 |

The ML model therefore reduces the evaluated operational cost by:

**€39,200**

or:

**9.83%**

compared with the 3-Sigma baseline.

The ML model also identified 105 problem gateways among its 105 selected gateways during the evaluated period, resulting in 100% precision for the selected visits in this evaluation.

---

## 11. Temporal Validation

A chronological train/validation split was used.

Training data:

- 2025-09-01 to 2026-01-05
- 5,247 training rows

Validation data:

- 2026-01-12 to 2026-03-23
- 3,229 validation rows

The model achieved:

- Accuracy: 90.99%
- Precision: 83.13%
- Recall: 92.63%
- F1: 87.62%
- ROC-AUC: 96.52%
- Average Precision: 95.12%

The temporal split is important because it evaluates the model on later observations rather than randomly mixing past and future records.

---

## 12. Unseen Gateway Evaluation

A separate gateway-level holdout experiment was performed.

The 320 gateways were divided into:

- 256 training gateways
- 64 completely unseen test gateways

The test gateways had zero overlap with the training gateways.

The model was trained only using the training gateways and evaluated on the unseen gateways during the later validation period.

Results:

- Accuracy: 91.74%
- Precision: 82.32%
- Recall: 89.07%
- F1: 85.56%
- ROC-AUC: 95.81%
- Average Precision: 93.39%

Gateway overlap:

**0**

This indicates that the model can generalise to gateways that were not present during model training.

---

## 13. Network Changes

Network behaviour is one of the strongest sources of predictive information.

The model uses recent and historical connectivity features so that it can detect deterioration.

The analysis showed substantial differences between normal and problem gateways.

For example:

| Feature | Normal | Problem |
|---|---:|---:|
| Recent 7-day disconnections | 14.23 | 278.49 |
| Recent 7-day disconnection hours | 12.83 | 68.80 |
| Recent 7-day offline hours | 4.24 | 200.78 |
| Recent 28-day disconnections | 65.59 | 1096.08 |
| Recent 28-day offline hours | 29.56 | 772.68 |

The trend features also show deterioration:

- Disconnection trend: -2.93 for normal gateways vs +5.56 for problem gateways.
- Offline-hours trend: -4.22 for normal gateways vs +10.08 for problem gateways.

Therefore, the model can respond to a gateway whose network behaviour is getting worse, rather than relying only on its long-term average behaviour.

---

## 14. Most Important Features

The model's strongest features were primarily related to recent connectivity behaviour.

The highest feature importances included:

1. Recent 7-day disconnections
2. Recent 7-day disconnection hours
3. Recent 28-day disconnections
4. Previous 21-day disconnections
5. Recent 28-day disconnection hours
6. Previous 21-day disconnection hours
7. Recent 7-day no-connection importance
8. Recent 28-day no-connection importance
9. Recent 7-day offline mean duration
10. Recent 7-day offline hours

This is operationally intuitive: repeated disconnections and increasing offline time are strong indicators of gateway instability.

---

## 15. Why Not Use the Engineer Review as the Main Target?

The engineer review dataset contains human judgements of "Normal" and "Schlecht" for 120 gateways.

It was not used as the primary training target.

The main prediction target was instead constructed from future operational behaviour using telemetry and meter-read outcomes.

This better matches the actual field-visit decision: whether a gateway is likely to experience a significant operational problem in the following week.

The engineer review can therefore remain useful as an additional qualitative validation source rather than defining the entire prediction problem.

---

## 16. Limitations

The solution has several limitations.

### Limited historical telemetry window

The detailed telemetry used for feature engineering covers August 2025 through March 2026. A longer historical period could provide more examples of seasonal and long-term behaviour.

### Target is an operational proxy

The problem definition is a designed operational proxy rather than an official failure label. Different thresholds could change the set of gateways considered problematic.

### Field visits are imperfect labels

A historical visit does not always mean that a gateway was actually broken. Some visits resulted in "no error found" or "no access".

### Cost evaluation is limited to labelled weeks

The 2026-03-23 prediction is produced, but its complete future seven-day outcome is unavailable in the supplied data, so it is not included in the cost comparison.

### Predictions are risk rankings, not guarantees

A high model probability means that a gateway is considered higher risk relative to other gateways. It does not guarantee that a gateway will fail.

### Network conditions can change unexpectedly

External events such as temporary connectivity failures, infrastructure changes, or environmental conditions may cause behaviour that was not represented in the historical training data.

### 15 visits remain a hard constraint

The model can rank gateways, but operations can still visit only 15 gateways per week. Some genuine problem gateways will therefore remain unvisited.

---

## 17. Reproducibility

The pipeline is designed to run from the project root and use the `data` directory by default.

Main components include:

- `baseline_3sigma.py`
- `src/build_features.py`
- `src/train_ml.py`
- `src/predict_ml.py`
- `src/evaluate_cost.py`
- `src/evaluate_unseen_gateways.py`
- `src/analyze_network_changes.py`
- `validate_submission.py`

The final prediction file is:

`predictions.csv`

It contains exactly 120 rows with 15 ranked gateways for each of the eight required weeks.

---

## 18. Final Decision

The ML approach was selected as the final solution because it satisfies the 15-visit operational constraint and achieved a lower evaluated operational cost than the provided 3-Sigma baseline.

On the seven fully evaluable weeks:

**3-Sigma baseline: €398,600**

**ML model: €359,400**

**Savings: €39,200 (9.83%)**

The model also demonstrated strong temporal performance and generalisation to completely unseen gateways.

For these reasons, the Random Forest risk-ranking approach is used to generate the final `predictions.csv`.