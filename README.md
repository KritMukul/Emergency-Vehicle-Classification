# Ambulance Detection Project

## Project Overview

An audio classification system to detect emergency vehicle sirens (ambulances, fire trucks, police) from regular traffic sounds using machine learning.

## Dataset

**Total Files:** ~9,988 audio samples

**Sources:**
- `data/audio/` - Urban Sound 8K dataset (Class 8 = Emergency vehicles)
- `data/Dataset/` - Custom emergency vehicle recordings (ambulance/firetruck/police)
- `data/traffic/` - Regular traffic sounds

## Features

52 audio features extracted per file:
- 13 MFCC (Mel-Frequency Cepstral Coefficients)
- 13 LFCC (Linear-Frequency Cepstral Coefficients)
- 13 CFCC (Constant-Q Cepstral Coefficients)
- 13 SFCC (Spectral-Frequency Cepstral Coefficients)

## Preprocessing

- **Sample Rate:** 22,050 Hz
- **Duration:** 3 seconds (66,150 samples)
- **Silence trimming** using librosa
- **Energy-based segmentation:** Extracts the loudest 3-second window (ensures emergency sirens are captured)
- **Zero-padding** for shorter audio files

## Machine Learning Models

### 1. Random Forest Classifier
- 200 estimators
- Parallel processing enabled
- Ensemble of decision trees with voting

### 2. LightGBM Classifier
- 300 estimators
- Learning rate: 0.05
- Gradient boosting for high accuracy
- Fast training with large datasets

### 3. XGBoost Classifier
- Extreme Gradient Boosting algorithm
- Optimized distributed gradient boosting
- Highly efficient and accurate
- Built-in regularization to prevent overfitting

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- EER (Equal Error Rate)

## How to Use

### Step 1: Install Dependencies

```bash
pip install librosa numpy pandas scikit-learn lightgbm xgboost scipy joblib tqdm
```

### Step 2: Extract Features

```bash
python preprocess.py
```

This will:
- Process all audio files from `data/audio/`, `data/Dataset/`, `data/traffic/`
- Extract 52 features per file
- Save features to:
  - `saved_features/*.npy` (NumPy arrays)
  - `features_csv/all_features.csv` (CSV file)

### Step 3: Train Models

**Option A - Random Forest:**
```bash
python train_randomforest.py
```

**Option B - LightGBM:**
```bash
python train_lightgbm.py
```

**Option C - XGBoost:**
```bash
python train_xgboost.py
```

All will:
- Load extracted features
- Split data 80/20 (train/test)
- Train the model
- Display performance metrics
- Save trained model as `.pkl` file

## Output Files

- `saved_features/X_mfcc.npy` - MFCC features
- `saved_features/X_lfcc.npy` - LFCC features
- `saved_features/X_cfcc.npy` - CFCC features
- `saved_features/X_sfcc.npy` - SFCC features
- `saved_features/y_labels.npy` - Labels (0=traffic, 1=emergency)
- `features_csv/all_features.csv` - All features combined
- `randomforest_siren_model.pkl` - Trained Random Forest model
- `lightgbm_siren_model.pkl` - Trained LightGBM model
- `xgboost_siren_model.pkl` - Trained XGBoost model

## Project Structure

```
Ambulance detection Diya/
├── preprocess.py - Feature extraction script
├── train_randomforest.py - Random Forest training
├── train_lightgbm.py - LightGBM training
├── train_xgboost.py - XGBoost training
├── data/
│   ├── audio/ - Urban Sound 8K dataset
│   ├── Dataset/ - Emergency vehicle recordings
│   └── traffic/ - Traffic sounds
├── features_csv/ - Extracted features (CSV)
├── saved_features/ - Extracted features (NumPy)
└── *.pkl - Trained models
```

## Key Innovation

**Energy-Based Audio Segmentation:**

Instead of simply taking the first 3 seconds, the system:
1. Slides a 3-second window across the entire audio
2. Calculates energy (sum of squared samples) for each window
3. Selects the window with **maximum energy**
4. **Result:** Always captures the loudest part (where sirens typically are)

## Technical Details

**Programming Language:** Python 3.12

**Core Libraries:**
- `librosa` - Audio processing
- `NumPy` - Numerical operations
- `pandas` - Data handling
- `scikit-learn` - Machine learning
- `LightGBM` - Gradient boosting
- `XGBoost` - Extreme gradient boosting
- `scipy` - Scientific computing
- `joblib` - Model serialization
- `tqdm` - Progress bars

## Applications

- Smart traffic management systems
- Autonomous vehicle detection
- Driver assistance systems
- Urban planning and emergency response analysis
- Assistive technology for hearing-impaired individuals

## Performance

Three models achieve high accuracy in distinguishing emergency vehicle sirens from regular traffic sounds:
- **Random Forest:** Strong baseline with ensemble voting
- **LightGBM:** Fast training with excellent performance
- **XGBoost:** Extreme gradient boosting with regularization

Specific metrics are displayed after training each model.


## Future Enhancements

- Real-time audio stream processing
- Deep learning models (CNN/RNN)
- Multi-class classification (distinguish ambulance vs fire truck vs police)
- Mobile deployment
- Distance and direction estimation
