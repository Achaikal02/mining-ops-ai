import pandas as pd
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

print("--- Memulai Investigasi Data Durasi ---")

try:
    df = pd.read_csv('hauling_activities.csv')
except FileNotFoundError:
    print("❌ ERROR: 'hauling_activities.csv' tidak ditemukan.")
    exit()

# --- 1. Konversi SEMUA kolom waktu ---
# Ini adalah daftar kolom waktu dari file hauling_activities
time_cols = [
    'queueStartTime', 'queueEndTime', 'loadingStartTime', 'loadingEndTime', 
    'departureTime', 'arrivalTime', 'dumpingStartTime', 'dumpingEndTime', 'returnTime'
]

# Ubah semua menjadi datetime, 'coerce' akan mengubah error menjadi NaT (Not a Time)
for col in time_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# --- 2. Hitung Ulang Durasi (Manual) ---
# Kita akan menghitung durasi kita sendiri dari timestamp
print("Menghitung ulang durasi dari timestamp...")

# Durasi Antri (menit)
df['queueDuration_calc'] = (df['queueEndTime'] - df['queueStartTime']).dt.total_seconds() / 60
# Durasi Muat (menit)
df['loadingDuration_calc'] = (df['loadingEndTime'] - df['loadingStartTime']).dt.total_seconds() / 60
# Durasi Angkut (menit)
df['haulingDuration_calc'] = (df['arrivalTime'] - df['departureTime']).dt.total_seconds() / 60
# Durasi Bongkar (menit)
df['dumpingDuration_calc'] = (df['dumpingEndTime'] - df['dumpingStartTime']).dt.total_seconds() / 60
# Durasi Kembali (menit)
df['returnDuration_calc'] = (df['returnTime'] - df['dumpingEndTime']).dt.total_seconds() / 60
# Total Waktu Siklus (menit)
df['totalCycleTime_calc'] = (df['returnTime'] - df['queueStartTime']).dt.total_seconds() / 60

# --- 3. Bandingkan dengan Data Asli ---
print("\n--- Perbandingan Durasi Asli vs. Perhitungan Ulang ---")

# Pilih kolom yang relevan untuk perbandingan
comparison_cols = [
    'queueDuration', 'queueDuration_calc',
    'loadingDuration', 'loadingDuration_calc',
    'haulingDuration', 'haulingDuration_calc',
    'totalCycleTime', 'totalCycleTime_calc'
]

# Tampilkan 10 baris pertama untuk perbandingan visual
print(df[comparison_cols].head(10))

# --- 4. Cek Perbedaan ---
# Hitung perbedaan absolut rata-rata
diff = (df['haulingDuration'] - df['haulingDuration_calc']).abs().mean()
print(f"\nPerbedaan rata-rata untuk 'haulingDuration': {diff:.2f} menit")

diff_total = (df['totalCycleTime'] - df['totalCycleTime_calc']).abs().mean()
print(f"Perbedaan rata-rata untuk 'totalCycleTime': {diff_total:.2f} menit")

if diff_total > 1.0: # Jika perbedaan rata-rata lebih dari 1 menit
    print("\nKESIMPULAN: 🔴 Data durasi asli TIDAK AKURAT.")
    print("Kita HARUS menggunakan 'totalCycleTime_calc' (dan durasi '_calc' lainnya) sebagai data target (y) kita.")
else:
    print("\nKESIMPULAN: 🟢 Data durasi asli terlihat AKURAT.")