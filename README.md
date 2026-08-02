# MediScope — Intelligent Multimodal System for Cardiovascular Disease Assessment

MediScope is an end-to-end computer vision + machine learning pipeline that analyzes echocardiogram (heart ultrasound) videos to automatically segment the left ventricle, track its motion across the cardiac cycle, and estimate **Ejection Fraction (EF)** — the key clinical metric used to diagnose heart failure and other cardiovascular conditions.

It combines **deep learning video segmentation** (YOLOv8) with **classical ML regression** to go from raw ultrasound video straight to a quantitative cardiac function score and a clinical risk category, on the full ~10,000-video [EchoNet-Dynamic](https://echonet.github.io/dynamic/) dataset.

## How it works

**1. Data preparation** — Loads EchoNet-Dynamic (10,030 echocardiogram videos + expert volume tracings), converts frame-level left-ventricle annotations into YOLO polygon labels, and splits into train/val/test (14,034 / 3,006 / 3,008 labeled frames).

**2. Left ventricle segmentation (YOLOv8n-seg)** — Trains a YOLOv8 instance segmentation model to detect and outline the left ventricle in each ultrasound frame.

**3. Ejection Fraction estimation** — Runs the trained model across full video sequences, tracks the left ventricle's mask area frame-by-frame, finds the End-Diastolic (ED, max area) and End-Systolic (ES, min area) frames, and computes:

```
EF = (ED_area − ES_area) / ED_area
```

A smoothed/optimized variant adds frame-skipping, mask smoothing, and resizing for speed and stability across all 10,030 videos.

<p align="center">
  <img src="images/lv_area_over_time.png" width="650" alt="Left ventricle area tracked across a cardiac cycle, with ED and ES frames marked">
</p>

*Left ventricle mask area tracked across a single cardiac cycle — the peak (ED) and trough (ES) frames are auto-detected and used to compute EF for that beat.*

**4. ML-based EF prediction** — Extracts statistical features from the area-over-time curve (max, min, mean, std, raw EF) and trains multiple regressors — Random Forest, Gradient Boosting, XGBoost, an LSTM, and an ensemble — automatically selecting the best by R².

**5. Clinical categorization** — Classifies each prediction into Low EF (<35%), Borderline (35–50%), or Normal (>50%), with a color-coded display and a suggested follow-up action for each category.

<p align="center">
  <img src="images/ef_clinical_display.png" width="350" alt="Color-coded EF clinical display">
</p>

## Results

**Segmentation (YOLOv8n-seg, validation set, 3,006 images):**

| Metric | Score |
|---|---|
| Precision | 0.998 |
| Recall | 0.998 |
| F1 | 0.998 |
| mAP@0.5 | 0.995 |
| mAP@0.5:0.95 | 0.785 |

**EF regression (test set, 2,006 samples):**

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **Gradient Boosting** ⭐ | 0.27% | 0.34% | 1.000 |
| XGBoost | 0.27% | 0.34% | 1.000 |
| Random Forest | 0.27% | 0.34% | 1.000 |
| Ensemble | 0.43% | 0.55% | 0.999 |
| LSTM | 1.35% | 1.76% | 0.989 |

<p align="center">
  <img src="images/model_comparison_chart.png" width="600" alt="Model comparison and statistical significance chart">
</p>

**Clinical category classification accuracy: 87%** (precision/recall of 0.87–0.93 across Low / Borderline / Normal EF categories).

> **A note on the numbers above:** the sub-1%-MAE / R²≈1.000 scores come from fitting on features engineered directly from the same segmentation-derived area curve used to define ground-truth EF, so they're close to a best-case, low-noise regression fit rather than a fully independent clinical validation. A separate calculation later in the notebook — comparing predictions against held-out video-level EF — reports a more realistic **MAE ≈ 3.9%, R² ≈ 0.92**, which is a more honest estimate of real-world generalization. Worth keeping both numbers in mind, and validating further before drawing clinical conclusions.

**Example: three patients across the EF spectrum**

<p align="center">
  <img src="images/three_patient_comparison.png" width="700" alt="Segmentation and EF comparison across low, borderline, and normal EF patients">
</p>

| Patient | EF | Status | Recommendation |
|---|---|---|---|
| A | 19.2% | 🔴 Severe impairment | Immediate cardiology referral |
| B | 37.8% | 🟡 Mild–moderate impairment | Follow-up in 1–3 months |
| C | 63.2% | 🟢 Normal function | Routine follow-up |

**Sample segmentation mask:**

<p align="center">
  <img src="images/segmentation_mask_example.png" width="350" alt="Example YOLOv8 segmentation mask on an echo frame">
</p>

## Tech stack

- **Computer vision:** Ultralytics YOLOv8 (instance segmentation), OpenCV
- **Machine learning:** scikit-learn (Random Forest, Gradient Boosting), XGBoost, TensorFlow/Keras (LSTM)
- **Data handling:** pandas, NumPy
- **Visualization:** Matplotlib, Seaborn

## Dataset

[EchoNet-Dynamic](https://echonet.github.io/dynamic/) (Stanford) — ~10,000 apical-4-chamber echocardiogram videos with expert-labeled EF values and left-ventricle tracings. Not included in this repo — download it from the link above or via Kaggle and set `DATA_ROOT` in the notebook.

## Repository contents

| File | Description |
|---|---|
| `MediScope-Cardiovascular-Disease.ipynb` | Full pipeline: data prep → YOLOv8 training → EF estimation → ML regression → clinical reporting |
| `images/` | Result figures used in this README |
| `requirements.txt` | Python dependencies |

## Getting started

```bash
git clone https://github.com/<your-username>/mediscope-cardiovascular-disease.git
cd mediscope-cardiovascular-disease
pip install -r requirements.txt
jupyter notebook MediScope-Cardiovascular-Disease.ipynb
```

Update `DATA_ROOT` at the top of the notebook to point to your local copy of EchoNet-Dynamic.

## Limitations

- EF regression metrics reflect performance on features derived from the model's own segmentation output — not an independent clinical trial. See the note above.
- Trained on EchoNet-Dynamic (a single-center, single-view dataset); generalization to other ultrasound machines, views, or patient populations is untested.
- This is a research/educational project, **not a certified diagnostic tool**. Outputs should not inform real clinical decisions without validation by a medical professional.

## License

Add a license of your choice (e.g. MIT) before publishing.
