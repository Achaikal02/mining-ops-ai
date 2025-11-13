import pandas as pd

try:
    df_full = pd.read_csv('hauling_activities.csv') 
    
    # --- PERBAIKAN: Filter HANYA untuk SHIFT_1 ---
    df = df_full[df_full['shift'] == 'SHIFT_1'].copy()
    
    if df.empty:
        print("❌ ERROR: Tidak ada data untuk 'SHIFT_1' di hauling_activities.csv.")
    else:
        print(f"--- Kalibrasi Presisi (HANYA SHIFT_1) ---")
        print(f"Menganalisis {len(df)} siklus dari SHIFT_1...")
        
        avg_loading = df['loadingDuration'].mean()
        avg_dumping = df['dumpingDuration'].mean()
        avg_return = df['returnDuration'].mean()
        avg_hauling = df['haulingDuration'].mean()
        
        print(f"Rata-rata 'loadingDuration' aktual: {avg_loading:.2f} menit")
        print(f"Rata-rata 'dumpingDuration' aktual: {avg_dumping:.2f} menit")
        print(f"Rata-rata 'returnDuration' aktual: {avg_return:.2f} menit")
        print(f"Rata-rata 'haulingDuration' aktual: {avg_hauling:.2f} menit")
        print("\nSilakan salin 4 angka BARU ini ke simulator.py Anda.")

except FileNotFoundError:
    print("❌ ERROR: 'hauling_activities.csv' tidak ditemukan.")
except Exception as e:
    print(f"❌ ERROR: {e}")