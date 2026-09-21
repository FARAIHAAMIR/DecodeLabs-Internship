import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ==========================================
# 1. DATASET SETUP & SIMULATION
# ==========================================
np.random.seed(42)
n_samples = 10000
n_fraud = int(n_samples * 0.0017)  # ~0.17% fraud rate

X_dummy = np.random.randn(n_samples, 10)
y_dummy = np.zeros(n_samples, dtype=int)
y_dummy[:n_fraud] = 1

df = pd.DataFrame(X_dummy, columns=[f'V{i}' for i in range(1, 11)])
df['Class'] = y_dummy

X = df.drop('Class', axis=1)
y = df['Class']

# ==========================================
# 2. STRATIFIED TRAIN-TEST SPLIT
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ==========================================
# 3. PIPELINE 1: LOGISTIC REGRESSION
# ==========================================
lr_pipeline = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

lr_param_grid = {
    'smote__k_neighbors': [3, 5],
    'classifier__C': [0.01, 0.1, 1.0]
}

print("Training Logistic Regression Pipeline with GridSearchCV...")
lr_grid = GridSearchCV(
    estimator=lr_pipeline,
    param_grid=lr_param_grid,
    scoring='roc_auc',
    cv=cv,
    n_jobs=-1
)
lr_grid.fit(X_train, y_train)

# ==========================================
# 4. PIPELINE 2: RANDOM FOREST
# ==========================================
rf_pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

rf_param_grid = {
    'smote__k_neighbors': [3, 5],
    'classifier__max_depth': [10, 20, None]
}

print("Training Random Forest Pipeline with GridSearchCV...")
rf_grid = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=rf_param_grid,
    scoring='roc_auc',
    cv=cv,
    n_jobs=-1
)
rf_grid.fit(X_train, y_train)

# ==========================================
# 5. MODEL EVALUATION
# ==========================================
def evaluate_model(model, name):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print(f"\n==========================================")
    print(f"EVALUATION REPORT: {name}")
    print(f"==========================================")
    print(f"Best Parameters: {model.best_params_}")
    print(f"ROC-AUC Score  : {roc_auc_score(y_test, y_proba):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report (Focus on Precision, Recall & F1):")
    print(classification_report(y_test, y_pred, zero_division=0))

evaluate_model(lr_grid, "Logistic Regression (with Scaler & SMOTE)")
evaluate_model(rf_grid, "Random Forest (with SMOTE)")
