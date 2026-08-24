# Offline Signature Verification using ResNet-50

A metric-learning-based offline handwritten signature verification project using **ResNet-50** to generate signature embeddings.

The project uses a strict **writer-disjoint experimental setup** and evaluates the trained model on the **CEDAR dataset** using multiple numbers of genuine reference signatures.

---

## 🚀 Project Highlights

- ResNet-50-based signature embedding model
- 256-dimensional L2-normalized embeddings
- Metric learning using **Contrastive Loss** and **Triplet Loss**
- Strict **writer-disjoint** train, validation, and test split
- 12,000 signature pairs
- 6,000 signature triplets
- CEDAR evaluation using **2, 3, 6, and 10 reference signatures**
- Evaluation metrics:
  - Accuracy
  - Balanced Accuracy
  - FAR
  - FRR
  - AUC
  - Threshold
- Validation-based threshold selection to avoid target-dataset threshold tuning

---

# 📂 Dataset Information

The project repository contains the following datasets:

| Dataset | Status |
|---|---|
| AAKASH | Included |
| CEDAR | Included |
| ROBIN / RENI | Included |

The dataset files are stored using **Git Large File Storage (Git LFS)**.

## Source Dataset Used for Training

The experimental setup combines the following source datasets:

| Dataset | Writers | Images |
|---|---:|---:|
| Akash Dataset | 686 | 14,626 |
| Robin Dataset | 64 | 4,298 |
| **Combined Dataset** | **750** | **18,924** |

The combined source dataset is used for model training, validation, and testing.

The **CEDAR dataset** is used for cross-dataset evaluation.

---

# 👥 Strict Writer-Disjoint Split

The combined source dataset is split at the **writer level**, rather than randomly splitting individual signature images.

| Split | Percentage | Writers |
|---|---:|---:|
| Training | 80% | 600 |
| Validation | 10% | 75 |
| Test | 10% | 75 |
| **Total** | **100%** | **750** |

The implementation verifies that no writer appears in more than one split:

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

This prevents **writer-level data leakage** and ensures that validation and test writers are unseen during training.

---

# 🔗 Metric Learning Samples

The training configuration uses:

| Sample Type | Count |
|---|---:|
| Signature Pairs | **12,000** |
| Signature Triplets | **6,000** |

## Pair Learning

Signature pairs are used for contrastive learning:

```text
Matching / Genuine Pair
        ↓
Smaller Embedding Distance


Non-Matching Pair
        ↓
Larger Embedding Distance
```

## Triplet Learning

Each triplet consists of:

```text
Anchor
├── Positive → Same Writer
└── Negative → Different Writer
```

The objective encourages:

```text
Distance(Anchor, Positive) < Distance(Anchor, Negative)
```

The combined training objective is:

```text
Total Loss = Contrastive Loss + 0.5 × Triplet Loss
```

---

# 🧠 Model Architecture

The system uses **ResNet-50** as the feature extraction backbone.

```text
Signature Image
       │
       ▼
   ResNet-50
       │
       ▼
Feature Representation
       │
       ▼
256-D Embedding
       │
       ▼
L2 Normalization
       │
       ▼
Signature Embedding
```

Verification is performed by comparing the embedding of a query signature with embeddings from genuine reference signatures.

---

# 🔬 CEDAR Zero-Shot Evaluation

The trained model is evaluated on the **CEDAR dataset** using different numbers of genuine reference signatures:

- 2 reference signatures
- 3 reference signatures
- 6 reference signatures
- 10 reference signatures

For each configuration, the verification threshold is selected using validation data and then applied to the CEDAR evaluation.

## CEDAR Evaluation Results

| Model | References | Accuracy | Balanced Accuracy | FAR ↓ | FRR ↓ | AUC ↑ | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| ResNet-50 | **2** | **79.76%** | **79.19%** | **7.58%** | **34.05%** | **0.9107** | **0.343098** |
| ResNet-50 | **3** | **81.54%** | **80.53%** | **4.39%** | **34.55%** | **0.9288** | **0.311824** |
| ResNet-50 | **6** | **85.32%** | **83.61%** | **4.39%** | **28.38%** | **0.9360** | **0.309600** |
| ResNet-50 | **10** | **87.42%** | **86.66%** | **10.45%** | **16.23%** | **0.9385** | **0.351160** |

---

# 🏆 Best Reported Configuration

The highest overall performance in the CEDAR evaluation was obtained using:

## 10 Reference Signatures

| Metric | Result |
|---|---:|
| Accuracy | **87.42%** |
| Balanced Accuracy | **86.66%** |
| AUC | **0.9385** |
| FAR | **10.45%** |
| FRR | **16.23%** |
| Threshold | **0.351160** |

---

# 📊 Performance Trend

Increasing the number of genuine reference signatures generally improved the overall verification performance.

| References | Accuracy | Balanced Accuracy | AUC |
|---:|---:|---:|
| 2 | 79.76% | 79.19% | 0.9107 |
| 3 | 81.54% | 80.53% | 0.9288 |
| 6 | 85.32% | 83.61% | 0.9360 |
| 10 | **87.42%** | **86.66%** | **0.9385** |

---

# 📏 Evaluation Metrics

## Accuracy

Overall percentage of correct verification decisions.

## Balanced Accuracy

Average classification performance across genuine and forged signature classes.

## FAR — False Acceptance Rate

Percentage of forged signatures incorrectly accepted as genuine.

**Lower is better.**

## FRR — False Rejection Rate

Percentage of genuine signatures incorrectly rejected.

**Lower is better.**

## AUC — Area Under the ROC Curve

Measures how effectively the model separates genuine and forged signatures across different thresholds.

**Higher is better.**

## Threshold

The verification decision boundary selected using validation data.

---

# 🔄 Experimental Pipeline

```text
        AAKASH DATASET
        686 Writers
       14,626 Images
              │
              │
              ▼
        ROBIN DATASET
         64 Writers
        4,298 Images
              │
              ▼
      COMBINED SOURCE DATA
      750 Writers / 18,924 Images
              │
              ▼
      WRITER-DISJOINT SPLIT
              │
      ┌───────┼────────┐
      ▼       ▼        ▼
    TRAIN    VALID    TEST
    80%      10%      10%
    600       75       75
   Writers  Writers  Writers
              │
              ▼
     12,000 Signature Pairs
              +
      6,000 Triplets
              │
              ▼
           ResNet-50
              │
              ▼
      256-D L2-Normalized
        Signature Embeddings
              │
              ▼
   Validation Threshold Selection
              │
              ▼
        Fixed Threshold
              │
              ▼
        CEDAR Evaluation
              │
      ┌───────┼────────┬────────┐
      ▼       ▼        ▼        ▼
    2 Ref   3 Ref    6 Ref    10 Ref
              │
              ▼
Accuracy / Balanced Accuracy
FAR / FRR / AUC
```

---

# 📁 Project Structure

```text
ResNet50-Signature-Verification_/
│
├── ResNet50_Only_2_3_6_10Ref.ipynb
├── README.md
├── requirements.txt
├── .gitignore
├── .gitattributes
│
└── data/
    ├── AAKASH/
    │   └── AAKASH.zip
    │
    ├── CEDAR/
    │   └── CEDAR.zip
    │
    └── ROBIN/
        └── ROBINRENI.zip
```

> **Note:** Large dataset files are managed using **Git LFS**.

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Tushar20250/ResNet50-Signature-Verification_.git
```

Move into the project directory:

```bash
cd ResNet50-Signature-Verification_
```

Install Git LFS and download the datasets:

```bash
git lfs install
git lfs pull
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then open and run:

```text
ResNet50_Only_2_3_6_10Ref.ipynb
```

---

# 📝 Summary

This project implements an **offline handwritten signature verification system** using **ResNet-50 and metric learning**.

The experimental setup uses:

```text
750 Writers
18,924 Signature Images

12,000 Signature Pairs
6,000 Signature Triplets

Strict Writer-Disjoint Split
80% Train / 10% Validation / 10% Test
```

The trained model is evaluated on the **CEDAR dataset** using multiple reference-signature configurations.

The best reported CEDAR result is:

```text
10 Reference Signatures

Accuracy:           87.42%
Balanced Accuracy:  86.66%
AUC:                 0.9385
FAR:                10.45%
FRR:                16.23%
```

---

# 👨‍💻 Author

**Tushar Kukreja**

B.Tech Computer Science Engineering

Offline Handwritten Signature Verification using ResNet-50
