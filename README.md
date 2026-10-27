# Orange Condition Classification — SVM vs MobileNetV2

**Scope note:** The supplied Google Drive dataset was an orange dataset, not an avocado dataset. This repository therefore classifies the provided orange labels and does not claim to solve avocado ripening-stage detection. The source labels are `blackspot`, `canker`, `fresh`, and `grenning` (spelling preserved).

##Google-colab link
https://colab.research.google.com/drive/1wPJXMeNacu3eiFJbRKuTmXOqwxGz0Msu?usp=sharing

## Recorded held-out test results (99 images)

| Model | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|
| SVM (HSV/LAB histograms + LBP) | 97.98% | 97.73% | 97.73% | 97.73% |
| MobileNetV2 | 92.93% | 93.01% | 92.05% | 92.37% |

SVM was stronger on this split. These results are preliminary; the test set is small and does not establish real-world performance.

## Structure
- `src/feature_extraction.py`: colour + LBP feature extraction
- `src/train_svm.py`: train and evaluate SVM
- `src/predict_image.py`: predict one image using SVM
- `src/train_mobilenet.py`: MobileNetV2 training template
- `outputs/model_comparison.csv`, `outputs/final_metrics.json`: recorded results
- `reports/final_report.pdf`: project report
- `notebooks/`: put exported Colab notebook here
- `models/`: copy trained artifacts from Colab here

## Dataset layout
Put the dataset at `data/orange_dataset/` with `train/<class>/` and `test/<class>/` folders. Dataset files are excluded from Git by default; do not publish them unless permitted.

## Setup
Python 3.10+ recommended:
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```
Train SVM:
```bash
python -m src.train_svm --data-dir data/orange_dataset/train --test-dir data/orange_dataset/test --output-dir models
```
Predict an image:
```bash
python -m src.predict_image --model models/orange_svm.joblib --image path/to/orange.jpg
```
MobileNetV2 training template:
```bash
python -m src.train_mobilenet --train-dir data/orange_dataset/train --output-dir models --epochs 5
```

## Limitations
Only 99 test images were used. Lighting, background, camera, fruit variety, and label bias may affect results. The four labels are not a complete ordered ripening scale. No production latency or field robustness was measured.
