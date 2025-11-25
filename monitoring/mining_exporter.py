import time
import pandas as pd
import os
from prometheus_client import start_http_server, Gauge

# --- 1. KONFIGURASI METRICS ---

# A. Metrics FUEL
FUEL_METRIC = Gauge('mining_fuel_consumption', 'Total konsumsi bahan bakar (Liter)', ['truck_id'])
EFFICIENCY_METRIC = Gauge('mining_fuel_efficiency', 'Efisiensi bahan bakar (Km/L)', ['truck_id'])

# B. Metrics EQUIPMENT STATUS
# Kita buat mapping: 1=Active, 2=Idle/Queue, 3=Maintenance, 4=Breakdown
EQUIPMENT_STATUS_METRIC = Gauge('mining_equipment_status_code', 'Status Alat (Code)', ['equipment_id', 'status_text'])

# C. Metrics PRODUCTION
PRODUCTION_METRIC = Gauge('mining_production_actual', 'Total Produksi (Ton)', ['site_id', 'shift_id'])

# Path Folder
DB_FOLDER = 'mining_db'

# Mapping Status Text ke Angka (Supaya Grafana bisa baca)
# Kita tambahkan 'IN_QUEUE' sesuai data kamu tadi
STATUS_MAPPING = {
    'ACTIVE': 1,
    'OPERATING': 1,
    'IDLE': 2,
    'IN_QUEUE': 2,    # Kita anggap antri sama dengan Idle
    'STANDBY': 2,
    'MAINTENANCE': 3,
    'BREAKDOWN': 4,
    'INACTIVE': 0
}

# --- FUNGSI 1: MEMPROSES FUEL (Tetap sama karena sudah benar) ---
def process_fuel():
    file_path = os.path.join(DB_FOLDER, 'fuel_consumption.csv')
    if not os.path.exists(file_path): return

    try:
        df = pd.read_csv(file_path)
        count = 0
        for _, row in df.iterrows():
            t_id = row.get('truckId')
            fuel = row.get('quantity')
            eff = row.get('fuelEfficiency')

            if pd.notna(t_id) and pd.notna(fuel):
                FUEL_METRIC.labels(truck_id=str(t_id)).set(float(fuel))
                if pd.notna(eff):
                    EFFICIENCY_METRIC.labels(truck_id=str(t_id)).set(float(eff))
                count += 1
        print(f"[FUEL] Update sukses: {count} data truk.")
    except Exception as e:
        print(f"[FUEL ERROR] {e}")

# --- FUNGSI 2: MEMPROSES EQUIPMENT STATUS (Revisi Logika ID & Status) ---
def process_equipment():
    file_path = os.path.join(DB_FOLDER, 'equipment_status_logs.csv')
    if not os.path.exists(file_path): return

    try:
        df = pd.read_csv(file_path)
        count = 0
        for _, row in df.iterrows():
            # 1. Cari ID Alat (Karena kolomnya terpisah-pisah)
            e_id = row.get('truckId')
            if pd.isna(e_id):
                e_id = row.get('excavatorId')
            if pd.isna(e_id):
                e_id = row.get('supportEquipmentId')
            
            # Jika semua ID kosong, skip baris ini
            if pd.isna(e_id):
                continue

            # 2. Ambil Status (Kolom 'currentStatus')
            status = row.get('currentStatus') 

            if pd.notna(status):
                # Ubah status jadi huruf besar & ganti spasi jadi underscore (jika ada)
                status_upper = str(status).upper().replace(' ', '_')
                
                # Mapping ke Angka
                status_code = STATUS_MAPPING.get(status_upper, 0) 

                # Kirim ke Prometheus
                EQUIPMENT_STATUS_METRIC.labels(
                    equipment_id=str(e_id), 
                    status_text=status_upper
                ).set(status_code)
                count += 1
        print(f"[EQUIPMENT] Update sukses: {count} status alat.")
    except Exception as e:
        print(f"[EQUIPMENT ERROR] {e}")

# --- FUNGSI 3: MEMPROSES PRODUCTION (Revisi Nama Kolom) ---
def process_production():
    file_path = os.path.join(DB_FOLDER, 'production_records.csv')
    if not os.path.exists(file_path): return

    try:
        df = pd.read_csv(file_path)
        count = 0
        for _, row in df.iterrows():
            # Revisi: Pakai 'miningSiteId' dan 'shift'
            site_id = row.get('miningSiteId')
            shift_id = row.get('shift')
            prod = row.get('actualProduction')

            if pd.notna(site_id) and pd.notna(prod):
                PRODUCTION_METRIC.labels(
                    site_id=str(site_id),
                    shift_id=str(shift_id)
                ).set(float(prod))
                count += 1
        print(f"[PRODUCTION] Update sukses: {count} record produksi.")
    except Exception as e:
        print(f"[PRODUCTION ERROR] {e}")

# --- MAIN LOOP ---
if __name__ == '__main__':
    print(f">>> Mining Exporter V5 (Final Fix) Berjalan!")
    start_http_server(8000)
    
    while True:
        print("\n--- Mengambil Data Baru ---")
        process_fuel()
        process_equipment()
        process_production()
        print("--- Selesai, tidur 15 detik ---")
        time.sleep(15)