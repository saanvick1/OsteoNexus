# OsteoNexus: An Attention-Driven Meta-Learning Framework for Osteoporosis Detection from Knee X-Rays

**Authors:** Saanvi Chakraborty (corresponding, chakrs12@gmail.com) and Manas Chakraborty\* (senior, mchakraborty82@gmail.com)

**Affiliation:** Mason Classical Academy, Naples, FL

\*Senior Author

---

## Overview

OsteoNexus is a lightweight, attention-driven meta-learning framework that detects osteoporosis from standard knee radiographs. The model combines three key architectural innovations — sigmoid attention gating, LSTM temporal modeling, and an autoencoder bottleneck — into a compact 88,746-parameter network that achieves 92.31% accuracy, 96.20% AUC, and 97.44% recall on an independent test set of 390 images.

The framework addresses a critical gap in osteoporosis screening: traditional diagnosis requires expensive DEXA scans, while knee X-rays are routinely acquired and widely available. OsteoNexus enables opportunistic osteoporosis screening from existing imaging studies without additional patient burden or cost.

## Key Results

| Metric | Value |
|--------|-------|
| Accuracy | 92.31% |
| Precision | 90.48% |
| Recall | 97.44% |
| F1-Score | 93.83% |
| AUC | 96.20% |
| ECE | 0.0372 |
| Brier Score | 0.0631 |
| Parameters | 88,746 |

**Confusion Matrix (Test Set, n=390):** TN=132, FP=24, FN=6, TP=228

**5-Fold Cross-Validation:** 95.12% +/- 0.65% accuracy, 98.03% +/- 0.48% AUC

**Multi-Seed Stability (seeds 42/123/456):** 91.88% +/- 0.87% accuracy

## Architecture

```
Input(200) -> BatchNorm -> Attention(sigmoid) -> Dense(128,ReLU) -> BN -> Dropout(0.3)
-> Dense(64,ReLU) -> Dropout(0.25) -> Reshape(1,64) -> LSTM(32) -> Dropout(0.25)
-> Encoder(16,ReLU) -> Dense(16,ReLU) -> Dropout(0.2) -> Softmax(2)
```

**Attention Gate:** Learns per-feature importance weights via sigmoid activation, focusing the model on the most discriminative PCA components.

**LSTM Layer:** Applies temporal gating (input/forget/output gates) as a learned nonlinear transformation on the intermediate representation.

**Autoencoder Bottleneck:** Compresses to a 16-dimensional latent space, forcing compact representation learning.

## Preprocessing Pipeline

Each knee X-ray undergoes a seven-step preprocessing pipeline:

1. **Grayscale loading and resizing** to 224x224 pixels with [0,1] normalization
2. **CLAHE contrast enhancement** (clipLimit=2.0, tileGridSize=8x8)
3. **Elastic transformation** (alpha=34, sigma=4) for data augmentation (training only)
4. **Gaussian blur** (5x5 kernel) for noise reduction
5. **Sobel/Scharr edge detection** with maximum response amalgamation
6. **Morphological dilation and erosion** with elliptical structuring elements (3x3)
7. **2D Fourier transform** with high-pass filtering to accentuate structural details

## Feature Extraction

Three complementary feature types are extracted and concatenated:

- **Pixel intensities:** 50,176 features (224x224 flattened)
- **HOG (Histogram of Oriented Gradients):** 26,244 features (9 orientations, 8x8 cells, 2x2 blocks)
- **LBP (Local Binary Patterns):** 26 features (24 points, radius 3, uniform method)
- **Total:** 76,446 features, reduced to **200 dimensions** via PCA (retaining 80.1% variance)

## Dataset

The [Multi-Class Knee Osteoporosis X-Ray Dataset](https://www.kaggle.com/datasets/mohamedgobara/multi-class-knee-osteoporosis-x-ray-dataset) (Gobara et al.) contains 1,947 knee radiographs annotated by orthopedic surgery specialists:

- Normal: 780 images (class 0)
- Osteopenia: 374 images (class 1 — At-Risk)
- Osteoporosis: 793 images (class 1 — At-Risk)

Binary mapping: Normal vs. At-Risk (Osteopenia + Osteoporosis)

## Statistical Validation

### DeLong Test (AUC Comparison)
| Baseline | z-statistic | p-value |
|----------|-------------|---------|
| Logistic Regression | 3.86 | <0.001 |
| SVM (RBF) | 3.29 | 0.001 |
| Random Forest | 3.07 | 0.002 |
| Gradient Boosting | 1.62 | 0.106 |

### McNemar Test (Error Comparison)
| Baseline | chi-squared | p-value |
|----------|-------------|---------|
| Logistic Regression | 22.04 | <0.001 |
| SVM (RBF) | 6.86 | 0.009 |
| Random Forest | 8.53 | 0.004 |
| Gradient Boosting | 5.04 | 0.025 |

## Project Structure

```
osteonexus/
  config.py          # Configuration, preprocessing pipeline, feature extraction
  model.py           # Model architecture, training, evaluation, statistical tests
  run.py             # Full 10-phase experimental pipeline and visualization

generate_revised_pdf.py          # JEI-formatted revised manuscript PDF generator
generate_figures_pdf.py          # JEI-formatted figures PDF (16 visualizations)
generate_response_to_reviewer.py # JEI-formatted response to reviewer comments
generate_code_pdf.py             # Annotated source code PDF generator

client/              # React/TypeScript research dashboard (frontend)
server/              # Express.js backend with PDF download routes
shared/              # Shared schema definitions
```

## Code Files

The three core model files are:

### `config.py`
Handles dataset loading, the complete seven-step image preprocessing pipeline, HOG/LBP/pixel feature extraction, PCA dimensionality reduction, and all hyperparameter configuration. Standardization is fit on training data only to prevent data leakage.

### `model.py`
Implements the OsteoNexus neural network architecture with configurable attention, LSTM, and autoencoder components. Includes training with class-balanced weights, early stopping, and learning rate scheduling. Also implements four baseline classifiers (Logistic Regression, SVM, Random Forest, Gradient Boosting), DeLong and McNemar statistical tests, bootstrap confidence intervals, calibration analysis, ablation study utilities, and two meta-learning algorithms (Reptile and Prototypical Networks).

### `run.py`
Orchestrates the complete experiment across ten sequential phases: data loading, feature extraction, model training, evaluation with statistical rigor, baseline comparison, statistical tests, component ablation, feature ablation, Reptile meta-learning, and Prototypical Network few-shot evaluation. Generates all publication-quality visualizations.

## Requirements

- Python 3.8+
- TensorFlow 2.x
- scikit-learn
- OpenCV (cv2)
- NumPy, Pandas, SciPy
- matplotlib, seaborn
- scikit-image
- tqdm
- ReportLab (for PDF generation)

## Usage

```bash
# Run the complete experimental pipeline
cd osteonexus
python run.py
```

Results (JSON metrics and PNG plots) are saved to `osteonexus/results/`.

## License

MIT License

Copyright (c) 2025 Saanvi Chakraborty and Manas Chakraborty

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
