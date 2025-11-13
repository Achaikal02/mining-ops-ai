import pandas as pd
import warnings
import numpy as np
import json 

# --- 1. Impor "Otak" Simulasi Anda ---
try:
    from simulator import run_hybrid_simulation, CONFIG, DB_TRUCKS, DB_EXCAVATORS, DB_ROADS
    print("✅ Mesin Simulasi Hybrid (dari simulator.py) berhasil dimuat.")
except ImportError as e:
    print(f"❌ ERROR: Gagal mengimpor dari 'simulator.py'. Pastikan file ada. Detail: {e}")
    exit()
except Exception as e:
    print(f"❌ ERROR saat memuat simulator.py: {e}")
    exit()

warnings.simplefilter(action='ignore', category=FutureWarning)

def run_backtest():
    print("\n--- Memulai Uji Backtesting ---")
    
    # --- 1. Muat Data Historis NYATA ---
    try:
        df_real = pd.read_csv('final_training_data_real.csv')
        print(f"Data historis nyata ('final_training_data_real.csv') dimuat. Total {len(df_real)} aktivitas.")
    except FileNotFoundError:
        print("❌ ERROR: 'final_training_data_real.csv' tidak ditemukan.")
        print("Pastikan 'create_training_data.py' sudah dijalankan.")
        return

    # --- 2. Pilih Periode Tes ---
    df_test_period = df_real[df_real['shift'] == 'SHIFT_1'].copy()
    if df_test_period.empty:
        print("❌ ERROR: Tidak ada data untuk 'SHIFT_1' di data historis.")
        return
        
    print(f"Memilih periode tes: 'SHIFT_1'. Ditemukan {len(df_test_period)} aktivitas hauling.")

    # --- 3. Hitung HASIL AKTUAL dari Data ---
    print("\n--- Menghitung Hasil AKTUAL (dari CSV) ---")
    
    harga_jual_ton = CONFIG['financial_params']['HargaJualBatuBara']
    harga_solar_liter = CONFIG['financial_params']['HargaSolar']
    
    actual_tonase = df_test_period['loadWeight'].sum()
    actual_bbm = df_test_period['fuelConsumed'].sum()
    actual_siklus = len(df_test_period)
    
    actual_pendapatan = actual_tonase * harga_jual_ton
    actual_biaya_bbm = actual_bbm * harga_solar_liter
    actual_profit_kotor = actual_pendapatan - actual_biaya_bbm
    
    print(f"  > Tonase Aktual: {actual_tonase:,.0f} ton")
    print(f"  > BBM Aktual: {actual_bbm:,.0f} liter")
    print(f"  > Jml Siklus Aktual: {actual_siklus} siklus")
    print(f"  > Profit Kotor Aktual: {actual_profit_kotor:,.0f} IDR")

    # --- 4. Tentukan KONDISI INPUT untuk Simulasi ---
    print("\n--- Menentukan Input Skenario (dari CSV) ---")
    
    input_skenario = {
        'weatherCondition': df_test_period['weatherCondition'].mode()[0],
        'roadCondition': df_test_period['roadCondition'].mode()[0],
        'shift': 'SHIFT_1',
        'alokasi_truk': df_test_period['truckId'].nunique(),
        'jumlah_excavator': df_test_period['excavatorId'].nunique(),
        'target_road_id': df_test_period['roadSegmentId'].mode()[0],
        'target_excavator_id': df_test_period['excavatorId'].mode()[0]
    }
    
    duration_hours = 8 
    
    print(f"Input untuk simulasi:")
    print(json.dumps(input_skenario, indent=2)) 

    # --- 5. Jalankan SIMULASI ---
    print(f"\n--- Menjalankan Simulasi Hybrid (Durasi {duration_hours} jam) ---")
    
    hasil_sim = run_hybrid_simulation(input_skenario, duration_hours=duration_hours)

    # --- 6. Ekstrak Hasil SIMULASI ---
    print("\n--- Menghitung Hasil SIMULASI ---")
    sim_tonase = hasil_sim['total_tonase']
    sim_bbm = hasil_sim['total_bbm_liter']
    sim_siklus = hasil_sim['jumlah_siklus_selesai']
    
    sim_pendapatan = sim_tonase * harga_jual_ton
    sim_biaya_bbm = sim_bbm * harga_solar_liter
    sim_profit_kotor = sim_pendapatan - sim_biaya_bbm
    
    print(f"  > Tonase Simulasi: {sim_tonase:,.0f} ton")
    print(f"  > BBM Simulasi: {sim_bbm:,.0f} liter")
    print(f"  > Jml Siklus Simulasi: {sim_siklus} siklus")
    print(f"  > Profit Kotor Simulasi: {sim_profit_kotor:,.0f} IDR")
    
    # --- 7. Tampilkan Laporan Perbandingan ---
    print("\n--- LAPORAN BACKTESTING (AKTUAL vs. SIMULASI) ---")
    
    def hitung_akurasi(aktual, simulasi):
        if aktual == 0: return 0.0
        return 100 * (1 - abs(aktual - simulasi) / aktual)

    print(f"| Metrik              | Aktual (dari CSV) | Simulasi (Hybrid) | Akurasi     |")
    print(f"|---------------------|-------------------|-------------------|-------------|")
    print(f"| Profit Kotor (IDR)  | {actual_profit_kotor:17,.0f} | {sim_profit_kotor:17,.0f} | {hitung_akurasi(actual_profit_kotor, sim_profit_kotor):>10.2f}% |")
    print(f"| Total Tonase (ton)  | {actual_tonase:17,.0f} | {sim_tonase:17,.0f} | {hitung_akurasi(actual_tonase, sim_tonase):>10.2f}% |")
    print(f"| Total BBM (liter)   | {actual_bbm:17,.0f} | {sim_bbm:17,.0f} | {hitung_akurasi(actual_bbm, sim_bbm):>10.2f}% |")
    print(f"| Jumlah Siklus       | {actual_siklus:17,.0f} | {sim_siklus:17,.0f} | {hitung_akurasi(actual_siklus, sim_siklus):>10.2f}% |")

if __name__ == "__main__":
    run_backtest()