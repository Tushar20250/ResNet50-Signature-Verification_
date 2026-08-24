# Offline Signature Verification using ResNet-50

A metric-learning based offline handwritten signature verification project using **ResNet-50** to generate signature embeddings. The project uses a strict **writer-disjoint experimental setup** and evaluates the trained model on the **CEDAR dataset** using multiple numbers of genuine reference signatures.

## Project Highlights

- ResNet-50 based signature embedding model
- 256-dimensional L2-normalized embeddings
- Metric learning using Contrastive Loss and Triplet Loss
- Strict writer-disjoint train / validation / test split
- 12,000 signature pairs
- 6,000 signature triplets
- CEDAR evaluation using 2, 3, 6, and 10 reference signatures
- Metrics: Accuracy, Balanced Accuracy, FAR, FRR, AUC, and Threshold
- Validation-based threshold selection to avoid target-dataset threshold tuning

---

# Dataset Status

The current project ZIP contains the following uploaded datasets:

| Dataset | Status |
|---|---|
| AAKASH | Included |
| CEDAR | Included |
| ROBIN | Empty folder reserved — add dataset later |
| RENI | Empty folder reserved — add dataset later |

## Source Dataset Used for Training

The experimental setup described in this project combines:

| Dataset | Writers | Images |
|---|---:|---:|
| Akash Dataset | 686 | 14,626 |
| Robin Dataset | 64 | 4,298 |
| **Combined Dataset** | **750** | **18,924** |

> **Note:** The current ZIP contains the uploaded AAKASH dataset and CEDAR dataset. Add the ROBIN and RENI dataset files into their respective folders when available.

---

# Strict Writer-Disjoint Split

The source dataset is split at the **writer level**, rather than randomly splitting individual images.

| Split | Percentage | Writers |
|---|---:|---:|
| Training | 80% | 600 |
| Validation | 10% | 75 |
| Test | 10% | 75 |
| **Total** | **100%** | **750** |

The intended validation checks are:

```python
assert set(train_df.writer_id).isdisjoint(val_df.writer_id)
assert set(train_df.writer_id).isdisjoint(test_df.writer_id)
assert set(val_df.writer_id).isdisjoint(test_df.writer_id)
```

Therefore:

```text
Train Writers ∩ Validation Writers = ∅
Train Writers ∩ Test Writers = ∅
Validation Writers ∩ Test Writers = ∅
```

This prevents writer-level data leakage and ensures unseen-writer evaluation.

---

# Metric Learning Samples

The training configuration uses:

| Sample Type | Count |
|---|---:|
| Signature Pairs | **12,000** |
| Signature Triplets | **6,000** |

## Pair Learning

Pairs are used for contrastive learning:

- Matching / genuine pairs → smaller embedding distance
- Non-matching pairs → larger embedding distance

## Triplet Learning

Each triplet contains:

```text
Anchor
├── Positive → Same Writer
└── Negative → Different Writer
```

The objective encourages:

```text
Distance(Anchor, Positive) < Distance(Anchor, Negative)
```

The combined objective is:

```text
Total Loss = Contrastive Loss + 0.5 × Triplet Loss
```

---

# CEDAR Zero-Shot Evaluation

The trained ResNet-50 model is evaluated on the **CEDAR dataset** using:

- 2 reference signatures
- 3 reference signatures
- 6 reference signatures
- 10 reference signatures

For each configuration, the verification threshold is selected using validation data and then applied for CEDAR evaluation.

The reported CEDAR results are:

| Model | References | Accuracy | Balanced Accuracy | FAR ↓ | FRR ↓ | AUC ↑ | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| ResNet-50 | **2** | **79.76%** | **79.19%** | **7.58%** | **34.05%** | **0.9107** | **0.343098** |
| ResNet-50 | **3** | **81.54%** | **80.53%** | **4.39%** | **34.55%** | **0.9288** | **0.311824** |
| ResNet-50 | **6** | **85.32%** | **83.61%** | **4.39%** | **28.38%** | **0.9360** | **0.309600** |
| ResNet-50 | **10** | **87.42%** | **86.66%** | **10.45%** | **16.23%** | **0.9385** | **0.351160** |

## Best Reported Configuration

**10 reference signatures**

- Accuracy: **87.42%**
- Balanced Accuracy: **86.66%**
- AUC: **0.9385**
- FAR: **10.45%**
- FRR: **16.23%**

---

# Evaluation Metrics

### Accuracy
Overall percentage of correct verification decisions.

### Balanced Accuracy
Average classification performance across genuine and forged classes.

### FAR — False Acceptance Rate
Percentage of forged signatures incorrectly accepted as genuine. Lower is better.

### FRR — False Rejection Rate
Percentage of genuine signatures incorrectly rejected. Lower is better.

### AUC
Measures the ability to separate genuine and forged signatures across decision thresholds. Higher is better.

### Threshold
The verification decision boundary selected using validation data.

---

# Experimental Pipeline

```text
AAKASH + ROBIN
      │
      ▼
Combined Source Dataset
750 Writers / 18,924 Images
      │
      ▼
Strict Writer-Disjoint Split
80% Train / 10% Validation / 10% Test
      │
      ▼
12,000 Pairs + 6,000 Triplets
      │
      ▼
ResNet-50
      │
      ▼
256-D L2-Normalized Embeddings
      │
      ▼
Validation Threshold Selection
      │
      ▼
CEDAR Evaluation
2 / 3 / 6 / 10 References
```

---

# Project Structure

```text
ResNet50_Signature_Verification_GitHub/
│
├── ResNet50_Only_2_3_6_10Ref.ipynb
├── README.md
├── requirements.txt
├── .gitignore
│
└── data/
    ├── AAKASH/
    ├── CEDAR/
    ├── ROBIN/   # Add dataset here later
    └── RENI/    # Add dataset here later
```

---

# Installation

```bash
pip install -r requirements.txt
```

Then open:

```bash
jupyter notebook
```

and run:

```text
ResNet50_Only_2_3_6_10Ref.ipynb
```

---

# Author

**Tushar Kukreja**

Offline Handwritten Signature Verification using ResNet-50
