import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

print("--- Memulai Investigasi Fitur ---")

try:
    # Gunakan file data gabungan yang sudah kita buat
    df = pd.read_csv('final_training_data_real.csv')
except FileNotFoundError:
    print("❌ ERROR: 'final_training_data_real.csv' tidak ditemukan.")
    print("Jalankan 'create_training_data.py' (dari langkah sebelumnya) terlebih dahulu.")
    exit()

# --- 1. Definisikan Fitur & Target yang Relevan ---
# Ambil dari train_models.py
numerical_features = [
    'capacity', 'bucketCapacity', 'rating', 'operator_experience_years',
    'distance', 'gradient', 'truck_age_days', 'days_since_last_maintenance'
]
categorical_features = [
    'weatherCondition', 'roadCondition', 'shift', 'brand', 'model_excavator'
]
targets = ['haulingDuration', 'queueDuration', 'fuelConsumed', 'loadWeight']

# Pastikan semua kolom ada dan bersih
all_cols = numerical_features + categorical_features + targets
df_clean = df[all_cols].dropna() # Hanya gunakan data yang lengkap untuk korelasi
df_clean[numerical_features] = df_clean[numerical_features].fillna(0)
df_clean[categorical_features] = df_clean[categorical_features].fillna('UNKNOWN')

if df_clean.empty:
    print("❌ ERROR: Tidak ada data yang lengkap untuk dianalisis.")
    exit()

# --- 2. Analisis Fitur Kategori (Variasi) ---
print("\n--- Analisis Variasi Fitur Kategori ---")
for col in categorical_features:
    print(f"\nDistribusi untuk '{col}':")
    print(df_clean[col].value_counts(normalize=True) * 100)
    print("-" * 20)
    
print("Periksa di atas: Jika ada satu nilai (misal: 'GOOD' 95%), fitur itu mungkin tidak berguna.")

# --- 3. Analisis Fitur Numerik (Korelasi) ---
print("\n--- Analisis Korelasi Fitur Numerik ---")
# Hitung korelasi antara semua fitur numerik dan target
correlation_matrix = df_clean[numerical_features + targets].corr()

# Fokus hanya pada korelasi dengan TARGETS
corr_with_targets = correlation_matrix[targets]
print(corr_with_targets)

print("\n--- KESIMPULAN KORELASI ---")
print("Angka yang mendekati +1.0 or -1.0 sangat bagus (prediktif).")
print("Angka yang mendekati 0.0 (misal: 0.05 atau -0.1) sangat buruk (tidak prediktif).")

# (Opsional) Visualisasikan heatmap
try:
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_with_targets, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Korelasi Fitur Numerik dengan Target")
    plt.tight_layout()
    # Simpan gambar
    plt.savefig('feature_correlation_heatmap.png')
    print("\n✅ Heatmap korelasi disimpan ke 'feature_correlation_heatmap.png'")
except Exception as e:
    print(f"\n(Gagal membuat heatmap: {e})")