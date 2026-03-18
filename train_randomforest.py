import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_curve
)
from sklearn.ensemble import RandomForestClassifier
import joblib

#  STEP 1: Load features 
df = pd.read_csv("features_csv/all_features.csv")
print(f"Loaded {len(df)} samples with {df.shape[1]-2} features.\n")

#  STEP 2: Prepare data 
X = df.drop(columns=["filename", "label"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#  STEP 3: Train Random Forest 
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

print("Training Random Forest model...\n")
model.fit(X_train, y_train)

#  STEP 4: Predictions 
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # probabilities for EER

#  STEP 5: Metrics 
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

# Compute EER (Equal Error Rate)
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
fnr = 1 - tpr
eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]

#  STEP 6: Print all results 
print("Model Performance Metrics:")
print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))
print("EER      :", round(eer, 4))

#  STEP 7: Save trained model 
joblib.dump(model, "randomforest_siren_model.pkl")
print("\nModel saved as 'randomforest_siren_model.pkl'")
