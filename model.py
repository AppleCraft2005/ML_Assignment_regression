import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.metrics import mean_absolute_percentage_error

dataset = pd.read_csv("dataset/gt_2015.csv")
target_y = "CO"
k = 5
zscore = 3.0

# x sbg fitur, y sbg target
x = dataset.drop(columns=["CO"])
y = dataset["CO"]

# zscore (hitung outlier existing)
y_mean = y.mean()
y_std = y.std(ddof=0)
z = (y - y_mean) / (y_std if y_std != 0 else 1.0)
outlier_mask = z.abs() > zscore
outlier_count = int(outlier_mask.sum())

print("Jumlah outlier:", outlier_count)

# kfold
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# model
model_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", LinearRegression())
])

model_huber = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", HuberRegressor())
])

# scoring
scoring = {
    "MAE": "neg_mean_absolute_error",
    "RMSE": "neg_root_mean_squared_error",
    "R2": "r2",
    "MAPE": "neg_mean_absolute_percentage_error"
}

def summarize_cv(results_dict):
    mae = -results_dict["test_MAE"]
    rmse = -results_dict["test_RMSE"]
    r2 = results_dict["test_R2"]
    mape = -results_dict["test_MAPE"]
    return {
        "MAE_mean": float(mae.mean()),
        "MAE_std": float(mae.std(ddof=1)),
        "RMSE_mean": float(rmse.mean()),
        "RMSE_std": float(rmse.std(ddof=1)),
        "R2_mean": float(r2.mean()),
        "R2_std": float(r2.std(ddof=1)),
        "MAPE_mean": float(mape.mean()),
        "MAPE_std": float(mape.std(ddof=1))
    }

# stress test outlier
add_outliers = 600
stress_factor = 5

def simple_outlier_stress(y, outlier_mask, add_n=add_outliers, factor=stress_factor):
    y_stress = y.copy()
    # indeks outlier lama
    existing_idx = np.where(outlier_mask)[0]
    # pilih tambahan random dari seluruh data
    random_idx = np.random.choice(len(y), size=add_n, replace=False)
    # gabungkan
    stressed_idx = np.unique(np.concatenate([existing_idx, random_idx]))
    # kalikan
    y_stress.iloc[stressed_idx] = y_stress.iloc[stressed_idx] * factor

    return y_stress, len(stressed_idx)


y_stress, stressed_total = simple_outlier_stress(
    y, outlier_mask, add_n=add_outliers, factor=stress_factor
)

#evaluasi
res_lr = cross_validate(model_lr, x, y, cv=kf, scoring=scoring)
sum_lr = summarize_cv(res_lr)

res_huber = cross_validate(model_huber, x, y, cv=kf, scoring=scoring)
sum_huber = summarize_cv(res_huber)

#evaluasi stress test
res_lr_stress = cross_validate(model_lr, x, y_stress, cv=kf, scoring=scoring)
sum_lr_stress = summarize_cv(res_lr_stress)

res_huber_stress = cross_validate(model_huber, x, y_stress, cv=kf, scoring=scoring)
sum_huber_stress = summarize_cv(res_huber_stress)

# =========================
# experiment log
# =========================
experiment_log = pd.DataFrame([
    {
        "ExpID": 1,
        "Target": target_y,
        "OutlierMethod": f"Z-Score on target (|z|>{zscore})",
        "OutlierCount": outlier_count,
        "Scaling": "StandardScaler",
        "Model": "LinearRegression",
        "Params": "default",
        "KFold": k,
        "MAE_mean": sum_lr["MAE_mean"],
        "MAE_std": sum_lr["MAE_std"],
        "RMSE_mean": sum_lr["RMSE_mean"],
        "RMSE_std": sum_lr["RMSE_std"],
        "MAPE_mean": sum_lr["MAPE_mean"],
        "MAPE_std": sum_lr["MAPE_std"],
        "R2_mean": sum_lr["R2_mean"],
        "R2_std": sum_lr["R2_std"],
        "Notes": "Baseline OLS"
    },
    {
        "ExpID": 2,
        "Target": target_y,
        "OutlierMethod": f"Z-Score on target (|z|>{zscore})",
        "OutlierCount": outlier_count,
        "Scaling": "StandardScaler",
        "Model": "HuberRegressor",
        "Params": "default",
        "KFold": k,
        "MAE_mean": sum_huber["MAE_mean"],
        "MAE_std": sum_huber["MAE_std"],
        "RMSE_mean": sum_huber["RMSE_mean"],
        "RMSE_std": sum_huber["RMSE_std"],
        "MAPE_mean": sum_huber["MAPE_mean"],
        "MAPE_std": sum_huber["MAPE_std"],
        "R2_mean": sum_huber["R2_mean"],
        "R2_std": sum_huber["R2_std"],
        "Notes": "Robust to outliers"
    },
    {
        "ExpID": 3,
        "Target": target_y,
        "OutlierMethod": f"Z-Score on target (|z|>{zscore}) + AddOutlier(n={add_outliers}) + Stress(y*{stress_factor})",
        "OutlierCount": stressed_total,
        "Scaling": "StandardScaler",
        "Model": "LinearRegression",
        "Params": "default",
        "KFold": k,
        "MAE_mean": sum_lr_stress["MAE_mean"],
        "MAE_std": sum_lr_stress["MAE_std"],
        "RMSE_mean": sum_lr_stress["RMSE_mean"],
        "RMSE_std": sum_lr_stress["RMSE_std"],
        "MAPE_mean": sum_lr_stress["MAPE_mean"],
        "MAPE_std": sum_lr_stress["MAPE_std"],
        "R2_mean": sum_lr_stress["R2_mean"],
        "R2_std": sum_lr_stress["R2_std"],
        "Notes": "Baseline OLS + Outlier Stress"
    },
    {
        "ExpID": 4,
        "Target": target_y,
        "OutlierMethod": f"Z-Score on target (|z|>{zscore}) + AddOutlier(n={add_outliers}) + Stress(y*{stress_factor})",
        "OutlierCount": stressed_total,
        "Scaling": "StandardScaler",
        "Model": "HuberRegressor",
        "Params": "default",
        "KFold": k,
        "MAE_mean": sum_huber_stress["MAE_mean"],
        "MAE_std": sum_huber_stress["MAE_std"],
        "RMSE_mean": sum_huber_stress["RMSE_mean"],
        "RMSE_std": sum_huber_stress["RMSE_std"],
        "MAPE_mean": sum_huber_stress["MAPE_mean"],
        "MAPE_std": sum_huber_stress["MAPE_std"],
        "R2_mean": sum_huber_stress["R2_mean"],
        "R2_std": sum_huber_stress["R2_std"],
        "Notes": "Huber + Outlier Stress"
    }
])

print(experiment_log.to_string(index=False))
experiment_log.to_csv("experiment_log.csv", index=False)
print("\nSaved: experiment_log.csv")