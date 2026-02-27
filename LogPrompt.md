# gimana cara membandingkan model regresi linear biasa dgn huber regresor di python

import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, cross_validate
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score

# =========================

# 1. LOAD DATASET

# =========================

# Ganti dengan dataset kamu

df = pd.read_csv("dataset.csv")

# Tentukan target

target_column = "target" # ganti sesuai nama target kamu

X = df.drop(columns=[target_column])
y = df[target_column]

# =========================

# 2. DEFINE K-FOLD

# =========================

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# =========================

# 3. DEFINE METRICS

# =========================

scoring = {
"MAE": "neg_mean_absolute_error",
"RMSE": "neg_root_mean_squared_error",
"R2": "r2"
}

# =========================

# 4. MODEL 1: Linear Regression (OLS)

# =========================

model_lr = Pipeline([
("scaler", StandardScaler()),
("model", LinearRegression())
])

results_lr = cross_validate(
model_lr, X, y,
cv=kf,
scoring=scoring,
return_train_score=False
)

# =========================

# 5. MODEL 2: Huber Regressor

# =========================

model_huber = Pipeline([
("scaler", StandardScaler()),
("model", HuberRegressor())
])

results_huber = cross_validate(
model_huber, X, y,
cv=kf,
scoring=scoring,
return_train_score=False
)

# =========================

# 6. PRINT RESULTS

# =========================

def print_results(name, results):
print(f"\n=== {name} ===")
print("MAE :", -results["test_MAE"].mean())
print("RMSE :", -results["test_RMSE"].mean())
print("R2 :", results["test_R2"].mean())

print_results("Linear Regression (OLS)", results_lr)
print_results("Huber Regressor", results_huber)

# lalu ingin menambahkan experiment log dari hasil evaluasi metrik tersebut, gimana

def run_and_log(exp_id, model_name, pipeline, params="default", notes=""):
results = cross_validate(
pipeline, X, y,
cv=kf,
scoring=scoring,
return_train_score=False
)

    # cross_validate untuk MAE/RMSE pakai NEGATIVE, jadi dibalik tanda (-)
    mae_scores = -results["test_MAE"]
    rmse_scores = -results["test_RMSE"]
    r2_scores = results["test_R2"]

    row = {
        "ExpID": exp_id,
        "Model": model_name,
        "Scaling": "StandardScaler",
        "Params": params,
        "KFold": kf.get_n_splits(),
        "MAE_mean": float(mae_scores.mean()),
        "MAE_std": float(mae_scores.std(ddof=1)),
        "RMSE_mean": float(rmse_scores.mean()),
        "RMSE_std": float(rmse_scores.std(ddof=1)),
        "R2_mean": float(r2_scores.mean()),
        "R2_std": float(r2_scores.std(ddof=1)),
        "Notes": notes
    }
    return row

experiment_log = []

experiment_log.append(
run_and_log(
exp_id=1,
model_name="LinearRegression",
pipeline=model_lr,
params="default",
notes="Baseline OLS"
)
)

experiment_log.append(
run_and_log(
exp_id=2,
model_name="HuberRegressor",
pipeline=model_huber,
params="default",
notes="Robust to outliers"
)
)

log_df = pd.DataFrame(experiment_log)
print(log_df.to_string(index=False))
log_df.to_csv("experiment_log.csv", index=False)
