import pandas as pd
import json
import warnings
import os

DATA_FOLDER = 'data'

# Nonaktifkan peringatan yang tidak relevan
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

print("--- Memulai Fase 1: Membuat Data Latih (Merging) ---")

try:
    # --- 1. Muat Semua Tabel ---
    print(f"Memuat 6 tabel utama dari folder '{DATA_FOLDER}'...")
    hauling = pd.read_csv(os.path.join(DATA_FOLDER, 'hauling_activities.csv'))
    trucks = pd.read_csv(os.path.join(DATA_FOLDER, 'trucks.csv'))
    excavators = pd.read_csv(os.path.join(DATA_FOLDER, 'excavators.csv'))
    operators = pd.read_csv(os.path.join(DATA_FOLDER, 'operators.csv'))
    roads = pd.read_csv(os.path.join(DATA_FOLDER, 'road_segments.csv'))
    maintenance = pd.read_csv(os.path.join(DATA_FOLDER, 'maintenance_logs.csv'))

    # --- 2. Bersihkan & Gabungkan (Join) Tabel ---
    print("Menggabungkan tabel (joining)...")
    
    # Ganti nama kolom 'id' di tabel konteks agar 'merge' lebih jelas
    trucks.rename(columns={'id': 'truckId'}, inplace=True)
    excavators.rename(columns={'id': 'excavatorId'}, inplace=True)
    operators.rename(columns={'id': 'operatorId'}, inplace=True)
    roads.rename(columns={'id': 'roadSegmentId'}, inplace=True)
    
    # Gabung 1: hauling + trucks
    df_merged = pd.merge(hauling, trucks, on='truckId', how='left', suffixes=('', '_truck'))
    
    # Gabung 2: df_merged + excavators
    df_merged = pd.merge(df_merged, excavators, on='excavatorId', how='left', suffixes=('', '_excavator'))
    
    # Gabung 3: df_merged + operators
    df_merged = pd.merge(df_merged, operators, on='operatorId', how='left', suffixes=('', '_operator'))
    
    # Gabung 4: df_merged + roads
    df_merged = pd.merge(df_merged, roads, on='roadSegmentId', how='left', suffixes=('', '_road'))

    print("Penggabungan 4 tabel konteks selesai.")

    # --- 3. Rekayasa Fitur (Feature Engineering) ---
    print("Memulai rekayasa fitur...")

    # A. Konversi Waktu (Sangat Penting)
    df_merged['timestamp'] = pd.to_datetime(df_merged['createdAt'])
    df_merged['truck_purchaseDate'] = pd.to_datetime(df_merged['purchaseDate'])
    
    # B. Hitung Fitur: Usia Truk
    df_merged['truck_age_days'] = (df_merged['timestamp'] - df_merged['truck_purchaseDate']).dt.days
    
    # C. Ekstrak Fitur: Pengalaman Operator (dari JSON)
    def extract_experience(json_str):
        try:
            return json.loads(json_str).get('years_experience', 0)
        except (json.JSONDecodeError, TypeError, AttributeError):
            return 0
            
    df_merged['operator_experience_years'] = df_merged['competency'].apply(extract_experience)

    # D. Fitur Maintenance (Kompleks)
    print("Memproses fitur maintenance (merge_asof)...")
    maint_logs_clean = maintenance.loc[maintenance['status'] == 'COMPLETED'].copy()
    maint_logs_clean['completionDate'] = pd.to_datetime(maint_logs_clean['completionDate'])
    
    # Gabungkan aktivitas hauling dengan log maintenance terakhirnya
    df_merged_maint = pd.merge_asof(
        df_merged.sort_values('timestamp'),
        maint_logs_clean[['truckId', 'completionDate']].sort_values('completionDate'),
        left_on='timestamp',
        right_on='completionDate',
        by='truckId',
        direction='backward'
    )
    
    # Hitung 'days_since_last_maintenance'
    df_merged_maint['days_since_last_maintenance'] = (df_merged_maint['timestamp'] - df_merged_maint['completionDate']).dt.days
    # Isi nilai NaN (truk baru) dengan asumsi 365 hari
    df_merged_maint['days_since_last_maintenance'].fillna(365, inplace=True)
    
    print("Rekayasa fitur selesai.")

    # --- 4. Simpan Hasil ---
    output_path = os.path.join(DATA_FOLDER, 'final_training_data_real.csv')
    df_merged_maint.to_csv(output_path, index=False)
    print(f"\n✅ DataFrame final telah disimpan ke '{output_path}'")

except FileNotFoundError as e:
    print(f"❌ ERROR: File tidak ditemukan. Pastikan '{e.filename}' ada di folder yang sama.")
except Exception as e:
    print(f"❌ ERROR Terjadi: {e}")