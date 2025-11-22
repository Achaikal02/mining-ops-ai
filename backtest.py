import pandas as pd
import warnings
import numpy as np
import json
import os

# --- 1. Impor "Otak" Simulasi & Konfigurasi ---
try:
    from simulator import run_hybrid_simulation, CONFIG, DB_TRUCKS, DB_EXCAVATORS, DB_ROADS
    print("✅ Mesin Simulasi Hybrid berhasil dimuat.")
except ImportError as e:
    print(f"❌ ERROR: Gagal mengimpor simulator: {e}")
    exit()

warnings.simplefilter(action='ignore', category=FutureWarning)

DATA_PATH = os.path.join('data', 'final_training_data_real.csv')

def hitung_akurasi(aktual, simulasi):
    """Rumus Akurasi: 100% - Persentase Error"""
    if aktual == 0: return 0.0
    return max(0, 100 * (1 - abs(aktual - simulasi) / aktual))

def run_backtest():
    print("\n--- Memulai Validasi Performa AI (Ekonomi & Fisika) ---")
    
    # 1. Muat Data
    try:
        df_real = pd.read_csv(DATA_PATH)
        df_real['queueStartTime'] = pd.to_datetime(df_real['queueStartTime'])
        df_real['returnTime'] = pd.to_datetime(df_real['returnTime'])
        df_real['date'] = df_real['queueStartTime'].dt.date
        print(f"Data historis dimuat. Total {len(df_real)} baris.")
    except FileNotFoundError:
        print(f"❌ ERROR: '{DATA_PATH}' tidak ditemukan.")
        return

    # 2. Pilih Sesi Terbaik (High Traffic untuk Stress Test)
    daily_counts = df_real.groupby(['date', 'shift']).size().reset_index(name='counts')
    if daily_counts.empty:
        print("Data kosong.")
        return
    
    # Urutkan dari yang PALING BANYAK aktivitasnya
    best_session = daily_counts.sort_values('counts', ascending=False).iloc[0]
    target_date = best_session['date']
    target_shift = best_session['shift']
    
    df_test = df_real[
        (df_real['date'] == target_date) & 
        (df_real['shift'] == target_shift)
    ].copy()
    
    print(f"\nSesi Validasi: {target_date} ({target_shift})")
    print(f"Jumlah Sampel Data: {len(df_test)} aktivitas")

    # 3. Hitung Metrik Performa AKTUAL (Benchmark)
    print("\n--- Menganalisis Data Lapangan (Aktual) ---")
    
    harga_solar = CONFIG['financial_params']['HargaSolar']
    
    # A. Fisika
    avg_ton_per_siklus_act = df_test['loadWeight'].mean()
    avg_waktu_siklus_act = df_test['totalCycleTime'].mean()
    
    # B. Ekonomi (Cost per Ton)
    total_tonase_act = df_test['loadWeight'].sum()
    total_bbm_act = df_test['fuelConsumed'].sum()
    total_biaya_bbm_act = total_bbm_act * harga_solar
    
    cost_per_ton_act = total_biaya_bbm_act / total_tonase_act

    print(f"  > Muatan Rata-rata   : {avg_ton_per_siklus_act:.2f} ton")
    print(f"  > Waktu Siklus Rata2 : {avg_waktu_siklus_act:.2f} menit")
    print(f"  > Cost BBM per Ton   : {cost_per_ton_act:,.0f} IDR/ton")
    
    # 4. Jalankan Simulasi Hybrid
    print("\n--- Menjalankan Simulasi Digital Twin ---")
    
    input_skenario = {
        'weatherCondition': df_test['weatherCondition'].mode()[0],
        'roadCondition': df_test['roadCondition'].mode()[0],
        'shift': target_shift,
        'alokasi_truk': df_test['truckId'].nunique(),
        'jumlah_excavator': df_test['excavatorId'].nunique(),
        'target_road_id': df_test['roadSegmentId'].mode()[0],
        'target_excavator_id': df_test['excavatorId'].mode()[0]
    }
    
    # Jalankan simulasi durasi panjang (24 jam) untuk mendapatkan rata-rata yang stabil
    hasil_sim = run_hybrid_simulation(input_skenario, CONFIG['financial_params'], duration_hours=24)
    
    # 5. Hitung Metrik Performa SIMULASI
    jml_siklus_sim = hasil_sim['jumlah_siklus_selesai']
    if jml_siklus_sim == 0:
        print("Simulasi gagal (0 siklus).")
        return

    # A. Fisika
    avg_ton_per_siklus_sim = hasil_sim['total_tonase'] / jml_siklus_sim
    
    # Hitung Effective Cycle Time Simulasi
    total_available_minutes = 24 * 60 * input_skenario['alokasi_truk']
    avg_waktu_siklus_sim = total_available_minutes / jml_siklus_sim

    # B. Ekonomi
    total_tonase_sim = hasil_sim['total_tonase']
    total_bbm_sim = hasil_sim['total_bbm_liter']
    total_biaya_bbm_sim = total_bbm_sim * harga_solar
    
    cost_per_ton_sim = total_biaya_bbm_sim / total_tonase_sim

    print(f"  > Muatan Rata-rata   : {avg_ton_per_siklus_sim:.2f} ton")
    print(f"  > Waktu Siklus Rata2 : {avg_waktu_siklus_sim:.2f} menit")
    print(f"  > Cost BBM per Ton   : {cost_per_ton_sim:,.0f} IDR/ton")

    # 6. Laporan Validasi Akhir
    print("\n==================================================")
    print("   LAPORAN VALIDASI AI (BUSINESS METRICS)         ")
    print("==================================================")
    print(f"| {'KPI (Key Performance Indicator)':<20} | {'Aktual':>10} | {'AI Model':>10} | {'Akurasi':>8} |")
    print(f"|----------------------|------------|------------|----------|")
    
    # Metrik 1: Produktivitas (Muatan)
    acc_ton = hitung_akurasi(avg_ton_per_siklus_act, avg_ton_per_siklus_sim)
    print(f"| {'Muatan (Ton/Trip)':<20} | {avg_ton_per_siklus_act:10.2f} | {avg_ton_per_siklus_sim:10.2f} | {acc_ton:>7.1f}% |")
    
    # Metrik 2: Kecepatan (Waktu)
    acc_time = hitung_akurasi(avg_waktu_siklus_act, avg_waktu_siklus_sim)
    print(f"| {'Cycle Time (Min)':<20} | {avg_waktu_siklus_act:10.2f} | {avg_waktu_siklus_sim:10.2f} | {acc_time:>7.1f}% |")
    
    # Metrik 3: Ekonomi (Cost)
    acc_cost = hitung_akurasi(cost_per_ton_act, cost_per_ton_sim)
    print(f"| {'Cost BBM (Rp/Ton)':<20} | {cost_per_ton_act:10.0f} | {cost_per_ton_sim:10.0f} | {acc_cost:>7.1f}% |")
    
    print("==================================================")
    
    # Kesimpulan Otomatis
    avg_acc = (acc_ton + acc_time + acc_cost) / 3
    print(f"\nKESIMPULAN MODEL:")
    if avg_acc > 90:
        print("⭐⭐⭐ PERFECT FIT. Model siap digunakan untuk keputusan strategis kritis.")
    elif avg_acc > 80:
        print("⭐⭐ GOOD FIT. Model sangat andal untuk perencanaan operasional.")
    elif avg_acc > 70:
        print("⭐ MODERATE FIT. Model menangkap tren utama, cukup untuk estimasi kasar.")
    else:
        print("⚠️ POOR FIT. Perlu kalibrasi ulang data durasi atau fitur ML.")

if __name__ == "__main__":
    run_backtest()