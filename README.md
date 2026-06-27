# ExoplanetDetector

An AI-powered exoplanet detection and classification pipeline using **NASA TESS light curve data**. The project preprocesses astronomical observations, detects potential exoplanet transits using the **Box Least Squares (BLS)** algorithm, estimates orbital parameters, and classifies detected signals into astrophysical categories using a **Conv1D + LSTM deep learning model**.

---

## Features

### 1. Preprocessing

* Load TESS FITS light curves
* Remove invalid observations
* Normalize stellar flux
* Detrend light curves using Wotan
* Save processed and detrended data

### 2. Transit Detection

* Box Least Squares (BLS) periodogram
* Automatic best transit candidate selection
* Batch processing of all detrended light curves
* Orbital period, duration, and epoch estimation
* Detection confidence estimation
* BLS periodogram visualization

### 3. AI Classification

* 5-class signal classification:
  * **Transit** — Confirmed exoplanet transits
  * **Eclipsing Binary** — Stellar binary systems
  * **Blend** — Blended/contaminated sources
  * **False Positive** — Instrumental or systematic artifacts
  * **Other** — Stellar variability, quiet stars
* Conv1D + Bidirectional LSTM deep learning architecture
* Label acquisition via NASA TOI catalog cross-matching
* Synthetic data augmentation for class balancing
* Per-candidate confidence scores and probability distributions
* Automated visualization of classified results

---

## Project Structure

```
ExoplanetDetector/
│
├── data/
│   ├── raw/                    # TESS FITS light curves
│   ├── detrended/              # Preprocessed .npz files
│   └── processed/              # Detection CSVs
│       ├── detection_results.csv
│       ├── candidates.csv
│       └── metadata.csv
│
├── preprocessing/              # Stage 1: Data preprocessing
│   ├── cleaner.py
│   ├── load_data.py
│   ├── normalizer.py
│   ├── detrender.py
│   ├── pipeline.py
│   └── visualize.py
│
├── detection/                  # Stage 2: Transit detection
│   ├── bls.py
│   ├── candidate.py
│   ├── parameters.py
│   ├── confidence.py
│   ├── pipeline.py
│   ├── batch_runner.py
│   └── visualize.py
│
├── classification/             # Stage 3: AI classification
│   ├── dataset.py              # Data loading, labeling, augmentation
│   ├── model.py                # Conv1D + LSTM architecture
│   ├── evaluate.py             # Metrics and plots
│   ├── utils.py                # Shared utilities
│   └── visualize.py            # Result visualization
│
├── model/                      # Saved model artifacts
│   ├── best_model.pth
│   ├── training_history.pkl
│   └── metrics.json
│
├── results/                    # Classification outputs
│   ├── classified_results.csv
│   ├── loss_curve.png
│   ├── accuracy_curve.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── classification_summary.png
│   └── figures/                # Per-candidate visualizations
│
├── utils/
│
├── main.py                     # Demo script (single file)
├── run_detection.py            # Run batch detection
├── train.py                    # Train the classifier
├── predict.py                  # Classify all candidates
├── config.py
└── README.md
```

---

## Complete Pipeline

### End-to-End Workflow

```
Raw FITS → Preprocessing → Detrended .npz → BLS Detection → Detection CSVs → AI Classification → Classified Results
```

### Stage 1: Preprocessing

```bash
python main.py
```

Processes raw TESS FITS files: cleaning, normalization, Wotan detrending → saves `.npz` files to `data/detrended/`.

### Stage 2: Transit Detection

```bash
python run_detection.py
```

Runs BLS on all detrended light curves → generates `detection_results.csv`, `candidates.csv`, and `metadata.csv` in `data/processed/`.

### Stage 3: Train the Classifier

```bash
python train.py
```

Automatically:
1. Loads all 99 detrended light curves
2. Acquires labels from the NASA TOI catalog + heuristic detection results
3. Augments the dataset to balance all 5 classes (100 samples each)
4. Builds a Conv1D + Bidirectional LSTM model (~570K parameters)
5. Trains with Adam optimizer, early stopping (patience=15), and LR scheduling
6. Evaluates on a held-out test set
7. Saves the best model, metrics, and all plots

**Output:**
```
model/best_model.pth           # Trained model weights
model/training_history.pkl     # Loss/accuracy per epoch
model/metrics.json             # Accuracy, F1, ROC-AUC, etc.
results/loss_curve.png         # Training/validation loss
results/accuracy_curve.png     # Training/validation accuracy
results/confusion_matrix.png   # Test set confusion matrix
results/roc_curve.png          # ROC curves per class
```

### Stage 4: Classify All Candidates

```bash
python predict.py
```

Automatically:
1. Loads the trained model
2. Reads every detrended light curve + detection results
3. Classifies each light curve into one of 5 categories
4. Generates per-candidate visualization figures
5. Saves `classified_results.csv`

**Output:**
```
results/classified_results.csv         # Full classification results
results/classification_summary.png     # Class distribution summary
results/figures/TIC_*_classification.png  # Per-candidate plots
```

---

## Model Architecture

```
Input (2048, 1)
  → Conv1D(64, k=7) → BatchNorm → ReLU → MaxPool(2)
  → Conv1D(128, k=5) → BatchNorm → ReLU → MaxPool(2)
  → Conv1D(256, k=3) → BatchNorm → ReLU → MaxPool(2)
  → Bidirectional LSTM(128)
  → Dropout(0.4)
  → Dense(128) → ReLU → Dropout(0.3)
  → Dense(5, Softmax)
```

**Hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Sequence Length | 2048 |
| Batch Size | 32 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Loss | Cross Entropy |
| Max Epochs | 100 |
| Early Stopping | Patience = 15 |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=7) |

---

## Output Files

### Detection Stage

| File | Description |
|------|-------------|
| `detection_results.csv` | BLS results for all 99 light curves |
| `candidates.csv` | Subset with detection power ≥ 0.003 |
| `metadata.csv` | Stellar parameters from FITS headers |

### Classification Stage

| File | Description |
|------|-------------|
| `classified_results.csv` | Predicted class + probabilities for every light curve |
| `metrics.json` | Test accuracy, precision, recall, F1, ROC-AUC |
| `best_model.pth` | Trained PyTorch model checkpoint |
| `training_history.pkl` | Epoch-by-epoch loss and accuracy |
| `loss_curve.png` | Training/validation loss plot |
| `accuracy_curve.png` | Training/validation accuracy plot |
| `confusion_matrix.png` | Test set confusion matrix |
| `roc_curve.png` | Per-class ROC curves with AUC |

### classified_results.csv Columns

| Column | Description |
|--------|-------------|
| `filename` | Source .npz filename |
| `tic_id` | TESS Input Catalog ID |
| `predicted_class` | Predicted astrophysical class |
| `confidence_score` | Model confidence (max probability) |
| `prob_transit` | Transit probability |
| `prob_eclipsingbinary` | Eclipsing binary probability |
| `prob_blend` | Blend probability |
| `prob_falsepositive` | False positive probability |
| `prob_other` | Other variability probability |

---

## Technologies Used

* Python 3
* PyTorch
* NumPy
* Pandas
* Scikit-learn
* Astropy
* Astroquery
* Wotan
* Matplotlib
* Seaborn

---

## Current Status

✅ Preprocessing Complete

✅ Transit Detection Complete

✅ AI Classification Complete

* Data Cleaning
* Flux Normalization
* Wotan Detrending
* BLS Periodogram
* Candidate Selection
* Parameter Estimation
* Confidence Estimation
* Batch Detection Processing
* TOI Catalog Cross-Matching
* Conv1D + LSTM Classification Model
* Model Training with Early Stopping
* Evaluation Metrics & Visualizations
* Batch Prediction Pipeline

---

## Dataset

NASA TESS (Transiting Exoplanet Survey Satellite) Light Curves — Sector 1

* 99 light curves (~18,000 data points each)
* Labels acquired from NASA Exoplanet Archive TOI catalog
* Dataset augmented with synthetic signal injection for class balancing

---

## Authors

* Amar Kumar Rajak
* Aditya Besra
