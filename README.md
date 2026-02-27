# Kelompok 2 Machine Learning 1

# 1. Muhammad Rizky (2310817310011)

# 2. Jovan Gilbert Natamasindah (2310817310002)

# 3. Devi Hafida Ariyani (2310817220018)

# Perbandingan Linear Regression vs Huber Regressor pada Prediksi Emisi CO

## Link Video Youtube

 [Youtube](https://youtu.be/pBcB7GQWvmU)

## Ringkasan

Proyek ini membandingkan dua model regresi untuk memprediksi variabel `CO` dari data turbin gas pada file `gt_2015.csv`:

- `LinearRegression` sebagai baseline.
- `HuberRegressor` sebagai model yang lebih robust terhadap outlier.

Evaluasi dilakukan menggunakan 5-Fold Cross Validation dan metrik `MAE`, `RMSE`, `MAPE`, serta `R2`. Hasil eksperimen dicatat otomatis ke `experiment_log.csv`.

## Struktur Proyek

- `model.py`: Skrip utama training, evaluasi, stress test outlier, dan logging hasil.
- `dataset/gt_2015.csv`: Dataset input.
- `experiment_log.csv`: Rekap hasil eksperimen.

## Dataset dan Target

Dataset berisi fitur operasi turbin gas seperti `AT`, `AP`, `AH`, `GTEP`, `TEY`, dll. Target prediksi adalah:

- `CO`

Pembagian variabel pada skrip:

- Fitur (`X`): semua kolom selain `CO`.
- Target (`y`): kolom `CO`.

## Metodologi

1. Deteksi outlier awal pada target `y` menggunakan Z-Score dengan ambang `|z| > 3.0`.
2. Siapkan dua pipeline model dengan `StandardScaler`.
3. Lakukan evaluasi baseline (data asli) dengan `KFold(n_splits=5, shuffle=True, random_state=42)`.
4. Lakukan stress test outlier dengan cara:
   - menambahkan indeks random (`add_outliers = 600`),
   - mengalikan nilai target terpilih dengan faktor `5`.
5. Evaluasi ulang kedua model pada data hasil stress test.
6. Simpan ringkasan mean dan std tiap metrik ke `experiment_log.csv`.

## Hasil Eksperimen

Berdasarkan `experiment_log.csv` yang ada di proyek saat ini:

- Kondisi normal (tanpa stress): `HuberRegressor` sedikit lebih baik pada `MAE` dan `MAPE`, dengan `R2` yang setara dengan `LinearRegression`.
- Kondisi stress outlier: `HuberRegressor` menunjukkan `MAE` dan `MAPE` yang jauh lebih baik dibanding baseline OLS.

## Kebutuhan Environment

Gunakan Python 3.9+ dengan library berikut:

- `pandas`
- `numpy`
- `scikit-learn`

## Cara Menjalankan

Dari root proyek, jalankan:

```bash
python model.py
```

Output utama:

- Jumlah outlier terdeteksi.
- Tabel hasil eksperimen di terminal.
- File `experiment_log.csv` yang terbarui.
