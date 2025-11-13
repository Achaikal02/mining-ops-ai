import pandas as pd
import joblib
import json
import warnings
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- DEFINISIKAN PATH ---
DATA_FOLDER = 'data'
MODEL_FOLDER = 'models'
# -------------------------

warnings.simplefilter(action='ignore', category=FutureWarning)

print(f"--- Memulai Pelatihan 3-Model (Upgrade V2.0) ---")

# --- 1. Muat Data "Matang" ---
try:
    input_path = os.path.join(DATA_FOLDER, 'final_training_data_real.csv')
    df = pd.read_csv(input_path)
    print(f"Berhasil memuat '{input_path}'.")
except FileNotFoundError:
    print("❌ ERROR: 'final_training_data_real.csv' tidak ditemukan.")
    print("Jalankan 'create_training_data.py' terlebih dahulu.")
    exit()

# --- 2. Definisikan Fitur (X) dan Target (y) ---
print("Mendefinisikan fitur (X) dan target (y)...")

# Fitur numerik (Input)
numerical_features = [
    'capacity', 'bucketCapacity', 'rating', 'operator_experience_years',
    'distance',  # Menggunakan 'distance' yang sudah divalidasi
    'gradient', 'truck_age_days', 'days_since_last_maintenance'
]

# Fitur kategori (Input)
categorical_features = [
    'weatherCondition', 'roadCondition', 'shift', 'brand', 'model_excavator'
]

# Target (Output yang ingin kita prediksi)
TARGET_FUEL = 'fuelConsumed'
TARGET_LOAD = 'loadWeight'
TARGET_DELAY = 'isDelayed' # <-- TARGET BARU KITA

# --- 3. Bersihkan Data ---
all_needed_cols = numerical_features + categorical_features + [TARGET_FUEL, TARGET_LOAD, TARGET_DELAY]
df_clean = df[all_needed_cols].dropna(subset=[TARGET_FUEL, TARGET_LOAD, TARGET_DELAY]) 
df_clean[numerical_features] = df_clean[numerical_features].fillna(0)
df_clean[categorical_features] = df_clean[categorical_features].fillna('UNKNOWN')
df_clean[TARGET_DELAY] = df_clean[TARGET_DELAY].astype(bool)

X = df_clean[numerical_features + categorical_features]
y_fuel = df_clean[TARGET_FUEL]
y_load = df_clean[TARGET_LOAD]
y_delay = df_clean[TARGET_DELAY]

print(f"Data latih siap dengan {len(X)} baris.")

# --- 4. Buat Preprocessing Pipeline ---
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ], remainder='drop')

# --- 5. Latih Model 1: Prediksi Konsumsi BBM (fuelConsumed) ---
print("\n--- Melatih Model 1: fuelConsumed ---")
pipeline_fuel = Pipeline(steps=[('preprocessor', preprocessor),
                                ('model', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
                               ])
X_train, X_test, y_train, y_test = train_test_split(X, y_fuel, test_size=0.2, random_state=42)
pipeline_fuel.fit(X_train, y_train)
# ... (logika evaluasi) ...
joblib.dump(pipeline_fuel, os.path.join(MODEL_FOLDER, 'model_fuel.joblib'))
print("✅ Model 'model_fuel.joblib' telah disimpan.")

# --- 6. Latih Model 2: Prediksi Berat Muatan (loadWeight) ---
print("\n--- Melatih Model 2: loadWeight ---")
pipeline_load = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('model', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
                                ])
X_train, X_test, y_train, y_test = train_test_split(X, y_load, test_size=0.2, random_state=42)
pipeline_load.fit(X_train, y_train)
# ... (logika evaluasi) ...
joblib.dump(pipeline_load, os.path.join(MODEL_FOLDER, 'model_load_weight.joblib'))
print("✅ Model 'model_load_weight.joblib' telah disimpan.")

# --- 7. Latih Model 3: Prediksi Probabilitas Delay (isDelayed) ---
print("\n--- Melatih Model 3: isDelayed (Classifier) ---")
pipeline_delay = Pipeline(steps=[('preprocessor', preprocessor),
                                ('model', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
                               ])
X_train, X_test, y_train, y_test = train_test_split(X, y_delay, test_size=0.2, random_state=42)
pipeline_delay.fit(X_train, y_train)
# ... (logika evaluasi) ...
joblib.dump(pipeline_delay, os.path.join(MODEL_FOLDER, 'model_delay_probability.joblib'))
print("✅ Model 'model_delay_probability.joblib' telah disimpan.")


# --- 8. Simpan Daftar Kolom Fitur ---
with open(os.path.join(MODEL_FOLDER, 'numerical_columns.json'), 'w') as f:
    json.dump(numerical_features, f)
with open(os.path.join(MODEL_FOLDER, 'categorical_columns.json'), 'w') as f:
    json.dump(categorical_features, f)

print(f"✅ Daftar kolom fitur telah disimpan ke '{MODEL_FOLDER}'.")
print("\n--- Pelatihan 3-Model Selesai ---")