# ExoplanetDetector

An AI-powered exoplanet detection pipeline using **NASA TESS light curve data**. The project preprocesses astronomical observations, detects potential exoplanet transits using the **Box Least Squares (BLS)** algorithm, estimates orbital parameters, computes a confidence score, and visualizes the detected transit signals.

---

## Features

### Preprocessing

* Load TESS FITS light curves
* Remove invalid observations
* Normalize stellar flux
* Detrend light curves using Wotan
* Save processed and detrended data

### Transit Detection

* Box Least Squares (BLS) periodogram
* Automatic best transit candidate selection
* Orbital period estimation
* Transit duration estimation
* Transit epoch estimation
* Detection confidence estimation
* BLS periodogram visualization

---

## Project Structure

```
ExoplanetDetector/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── detrended/
│
├── preprocessing/
│   ├── cleaner.py
│   ├── load_data.py
│   ├── normalizer.py
│   ├── detrender.py
│   └── pipeline.py
│
├── detection/
│   ├── bls.py
│   ├── candidate.py
│   ├── parameters.py
│   ├── confidence.py
│   ├── visualize.py
│   └── pipeline.py
│
├── utils/
│
├── main.py
├── config.py
└── README.md
```

---

## Detection Pipeline

1. Load detrended light curve
2. Run Box Least Squares (BLS)
3. Detect strongest transit signal
4. Select the best candidate
5. Estimate orbital parameters
6. Compute confidence score
7. Visualize the BLS periodogram

---

## Example Output

```
========== BLS RESULT ==========
Best Period      : 6.8115 days
Transit Duration : 0.0250 days
Transit Epoch    : 1328.4837
Detection Power  : 0.0047

========== CANDIDATE ==========
Best transit candidate selected.

========== PARAMETERS ==========
Orbital Period   : 6.8115 days
Transit Duration : 0.0250 days
Transit Epoch    : 1328.4837

========== CONFIDENCE ==========
Confidence Score : 46.89%
```

---

## Technologies Used

* Python 3
* NumPy
* Astropy
* Wotan
* Matplotlib

---

## Current Status

✅ Preprocessing Complete

✅ Transit Detection Complete

* Data Cleaning
* Flux Normalization
* Wotan Detrending
* BLS Periodogram
* Candidate Selection
* Parameter Estimation
* Confidence Estimation
* Visualization

---

## Future Work

* Machine Learning based candidate classification
* Planet radius estimation
* Stellar parameter estimation
* False positive filtering
* Multi-transit validation
* Interactive visualization dashboard

---

## Dataset

NASA TESS (Transiting Exoplanet Survey Satellite) Light Curves

---

## Authors

* Amar Kumar Rajak
* Aditya Besra
