import pandas as pd

try:
    df_full = pd.read_csv('hauling_activities.csv') 
    
    # --- Filter HANYA untuk SHIFT_1 ---
    df = df_full[df_full['shift'] == 'SHIFT_1'].copy()
    
    if df.empty:
        print("❌ ERROR: Tidak ada data untuk 'SHIFT_1'.")
    else:
        print(f"--- Validasi Durasi SHIFT_1 ---")
        print(f"Menganalisis {len(df)} siklus dari SHIFT_1...")
        
        # Konversi waktu
        df['queueStartTime'] = pd.to_datetime(df['queueStartTime'])
        df['returnTime'] = pd.to_datetime(df['returnTime'])
        
        # Temukan aktivitas pertama dan terakhir
        first_activity_start = df['queueStartTime'].min()
        last_activity_end = df['returnTime'].max()
        
        # Hitung durasi total
        total_duration = last_activity_end - first_activity_start
        total_duration_hours = total_duration.total_seconds() / 3600.0
        
        print(f"\nAktivitas Pertama Dimulai: {first_activity_start}")
        print(f"Aktivitas Terakhir Selesai: {last_activity_end}")
        print(f"Total Durasi Aktual: {total_duration_hours:.2f} jam")
        
        print(f"\nTotal Siklus Aktual: {len(df)} siklus")
        
        # Hitung rata-rata siklus per jam (produktivitas nyata)
        avg_cycles_per_hour = len(df) / total_duration_hours
        print(f"Produktivitas Aktual: {avg_cycles_per_hour:.2f} siklus/jam")

except FileNotFoundError:
    print("❌ ERROR: 'hauling_activities.csv' tidak ditemukan.")
except Exception as e:
    print(f"❌ ERROR: {e}")