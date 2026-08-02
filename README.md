# MediScope — Intelligent Multimodal System for Cardiovascular Disease Assessment

MediScope is an end-to-end computer vision + machine learning pipeline that analyzes echocardiogram (heart ultrasound) videos to automatically segment the left ventricle, track its motion across the cardiac cycle, and estimate **Ejection Fraction (EF)** — a key clinical indicator used to diagnose heart failure and other cardiovascular conditions.

The project combines **deep learning-based video segmentation** with **classical ML regression** to move from raw ultrasound video → quantitative cardiac function metrics, mimicking part of the workflow a cardiologist performs manually.

## How it works

The pipeline is built in four stages:

1. **Data preparation** — Loads the [EchoNet-Dynamic](https://echonet.github.io/dynamic/) dataset (echocardiogram videos + expert volume tracings), converts frame-level ventricle annotations into YOLO-format polygon labels, and splits videos into train/val/test sets.

2. **Left ventricle segmentation (YOLOv8-seg)** — Trains a YOLOv8 instance segmentation model to detect and outline the left ventricle in each ultrasound frame, then evaluates it with precision, recall, F1, and mAP metrics.

3. **Ejection Fraction estimation** — Runs the trained segmentation model across full video sequences, tracks the left ventricle's mask area frame by frame, identifies the End-Diastolic (ED, max area) and End-Systolic (ES, min area) frames, and computes EF as:

   ```
   EF = (ED_area − ES_area) / ED_area
   ```

   Includes a smoothed/optimized variant (`segment_video_medical`) with frame-skipping, mask smoothing, and resizing for faster, more stable estimation on full-length videos.

4. **ML-based prediction** — Extracts statistical features from the area-over-time curve (max, min, mean, std, raw EF) and trains regression models (Random Forest, Gradient Boosting, XGBoost, and a Keras neural network) to predict EF, automatically selecting the best-performing model by R² score. Includes a color-coded clinical display (normal / borderline / low EF) for interpreting results.

## Tech stack

- **Computer vision:** Ultralytics YOLOv8 (segmentation), OpenCV
- **Machine learning:** scikit-learn (Random Forest, Gradient Boosting), XGBoost, TensorFlow/Keras
- **Data handling:** pandas, NumPy
- **Visualization:** Matplotlib, Seaborn

## Dataset

Uses the **EchoNet-Dynamic** dataset (Stanford), which contains ~10,000 apical-4-chamber echocardiogram videos with expert-labeled EF values and left ventricle tracings. The dataset is not included in this repo — download it from [echonet.github.io/dynamic](https://echonet.github.io/dynamic/) or via Kaggle and update the `DATA_ROOT` path in the notebook.

## Repository contents

| File | Description |
|---|---|
| `MediScope-Cardiovascular-Disease.ipynb` | Full pipeline: data prep → YOLOv8 training → EF estimation → ML regression |
| `requirements.txt` | Python dependencies |

## Getting started

```bash
git clone https://github.com/<your-username>/mediscope-cardiovascular-disease.git
cd mediscope-cardiovascular-disease
pip install -r requirements.txt
jupyter notebook MediScope-Cardiovascular-Disease.ipynb
```

Update the dataset path (`DATA_ROOT`) at the top of the notebook to point to your local copy of EchoNet-Dynamic.

## Notes

- Originally developed and trained on Kaggle (GPU-accelerated environment).
- This is a research/educational project, not a certified diagnostic tool — outputs should not be used for real clinical decision-making without validation by a medical professional.

## License

Add a license of your choice (e.g. MIT) before publishing.
