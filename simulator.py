import pandas as pd
import joblib
import json
import warnings
import numpy as np
import ollama
import os
import simpy
from itertools import product

# --- 0. Nonaktifkan Peringatan ---
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# --- DEFINISIKAN PATH ---
DATA_FOLDER = 'data'
MODEL_FOLDER = 'models'
# -------------------------

# --- 1. DEFINISI FUNGSI UTAMA ---

def load_config():
    print("Memuat Konfigurasi Finansial (Hardcoded)...")
    return {
        'financial_params': {
            'HargaJualBatuBara': 800000, 
            'HargaSolar': 15000, 
            'BiayaPenaltiKeterlambatanKapal': 100000000,
            'BiayaRataRataInsiden': 50000000
        }
    }

def get_features_for_prediction(truck_id, operator_id, road_id, excavator_id, weather, road_cond, shift):
    try:
        truck = DB_TRUCKS.loc[truck_id]
        excavator = DB_EXCAVATORS.loc[excavator_id]
        operator = DB_OPERATORS.loc[operator_id]
        road = DB_ROADS.loc[road_id]
        
        truck_age_days = (pd.Timestamp.now(tz='UTC') - pd.to_datetime(truck['purchaseDate'])).days
        
        try:
            op_exp = json.loads(operator['competency']).get('years_experience', 0)
        except:
            op_exp = 0
            
        today = pd.Timestamp.now(tz='UTC')
        last_maint = DB_MAINTENANCE_SORTED.loc[
            (DB_MAINTENANCE_SORTED['truckId'] == truck_id) &
            (DB_MAINTENANCE_SORTED['completionDate'] < today)
        ]
        
        days_since_maint = 365 
        if not last_maint.empty:
            days_since_maint = (today - last_maint.iloc[-1]['completionDate']).days

        feature_dict = {
            'capacity': truck['capacity'],
            'bucketCapacity': excavator['bucketCapacity'],
            'rating': operator['rating'],
            'operator_experience_years': op_exp,
            
            # --- PERBAIKAN BUG UTAMA DI SINI ---
            'distance': road['distance'], # Kunci harus 'distance' agar cocok dengan Model ML
            # -----------------------------------
            
            'gradient': road['gradient'],
            'truck_age_days': truck_age_days,
            'days_since_last_maintenance': days_since_maint,
            'weatherCondition': weather,
            'roadCondition': road_cond,
            'shift': shift,
            'brand': truck['brand'],
            'model_excavator': excavator['model']
        }
        
        return pd.DataFrame([feature_dict])[MODEL_COLUMNS]

    except KeyError as e:
        # print(f"Debug: Data tidak ditemukan {e}") 
        return pd.DataFrame(columns=MODEL_COLUMNS)
    except Exception as e:
        # print(f"Debug: Error fitur {e}")
        return pd.DataFrame(columns=MODEL_COLUMNS)

def truck_process_hybrid(env, truck_id, operator_id, resources, global_metrics, skenario):
    """
    'Kehidupan' satu truk (SimPy + ML).
    """
    weather = skenario['weatherCondition']
    road_cond = skenario['roadCondition']
    shift = skenario['shift']
    excavator_resource = resources['excavator']
    
    excavator_id = skenario.get('target_excavator_id')
    road_id = skenario.get('target_road_id')
    
    try:
        truck_data = DB_TRUCKS.loc[truck_id]
        kapasitas_ton = truck_data['capacity']
    except KeyError:
        return 

    # Logika Faktor Cuaca
    weather_factor = 1.0
    if "Hujan" in str(weather):
        weather_factor = 1.25
    elif "Licin" in str(road_cond):
        weather_factor = 1.15

    while True:
        X_features = get_features_for_prediction(
            truck_id, operator_id, road_id, excavator_id, weather, road_cond, shift
        )
        
        # Default Values
        fuel_consumed_pred = 10.0 
        load_weight_pred = kapasitas_ton * 0.87
        delay_proba_pred = 0.0
        
        if not X_features.empty:
            try:
                fuel_consumed_pred = MODEL_FUEL.predict(X_features)[0]
                raw_load = MODEL_LOAD.predict(X_features)[0]
                load_weight_pred = raw_load * 0.87
                delay_proba_pred = MODEL_DELAY.predict_proba(X_features)[0][1]
            except Exception:
                pass
        
        # --- SIMULASI FISIK ---
        # 1. Hauling
        avg_hauling_menit = 31.76
        actual_hauling = avg_hauling_menit * weather_factor
        yield env.timeout(actual_hauling / 60.0)
        
        # 2. Queue & Loading
        with excavator_resource.request() as req:
            yield req 
            # Masuk Loading
            avg_loading_menit = 11.02
            actual_loading = avg_loading_menit * (1.1 if "Hujan" in str(weather) else 1.0)
            yield env.timeout(actual_loading / 60.0)
        
        # 3. Return
        avg_return_menit = 25.29
        actual_return = avg_return_menit * weather_factor
        yield env.timeout(actual_return / 60.0)
        
        # 4. Dumping
        avg_dumping_menit = 8.10
        yield env.timeout(avg_dumping_menit / 60.0)
        
        # --- PENCATATAN ---
        global_metrics['total_tonase'] += load_weight_pred 
        
        # BBM dipengaruhi cuaca juga (asumsi RPM tinggi / slip)
        adjusted_fuel = fuel_consumed_pred * weather_factor
        global_metrics['total_bbm_liter'] += (adjusted_fuel * 1.6) 
        
        global_metrics['total_probabilitas_delay'] += delay_proba_pred
        
        # Hitung waktu antri (Total Waktu - Waktu Gerak Fisik)
        # (SimPy otomatis menghandle waktu tunggu di 'yield req')
        # Kita bisa menambah counter antrian di sini jika mau lebih detail,
        # tapi untuk profit, 'total_waktu_simulasi' yang memanjang otomatis mengurangi siklus.
        
        global_metrics['jumlah_siklus_selesai'] += 1

def run_hybrid_simulation(skenario, financial_params, duration_hours=24):
    env = simpy.Environment()
    
    jumlah_excavator_aktif = skenario.get('jumlah_excavator', 1)
    resources = {'excavator': simpy.Resource(env, capacity=jumlah_excavator_aktif)}
    
    global_metrics = {
        'total_tonase': 0, 'total_bbm_liter': 0, 
        'jumlah_siklus_selesai': 0, 'total_probabilitas_delay': 0.0
    }
    
    all_trucks = DB_TRUCKS.index.tolist()
    all_operators = DB_OPERATORS.index.tolist()
    
    for i in range(skenario['alokasi_truk']):
        truck_id = all_trucks[i % len(all_trucks)]
        operator_id = all_operators[i % len(all_operators)]
        
        skenario_lengkap = skenario.copy()
        skenario_lengkap['target_excavator_id'] = skenario.get('target_excavator_id', DB_EXCAVATORS.index[0])
        skenario_lengkap['target_road_id'] = skenario.get('target_road_id', DB_ROADS.index[0])
        
        env.process(truck_process_hybrid(env, truck_id, operator_id, resources, global_metrics, skenario_lengkap))

    env.run(until=duration_hours)
    
    # --- Hitung Profit ---
    harga_jual = financial_params['HargaJualBatuBara']
    harga_solar = financial_params['HargaSolar']
    biaya_insiden = financial_params.get('BiayaRataRataInsiden', 50000000)
    # Kita bisa tambah biaya sewa per jam jika mau
    
    pendapatan = global_metrics['total_tonase'] * harga_jual
    biaya_bbm = global_metrics['total_bbm_liter'] * harga_solar
    biaya_risiko_insiden = global_metrics['total_probabilitas_delay'] * biaya_insiden
    
    # Biaya Antrian implisit: Antrian = Siklus Lebih Sedikit = Pendapatan Lebih Kecil.
    # Jadi kita tidak perlu double counting biaya antrian kecuali ada denda khusus.
    
    profit = pendapatan - biaya_bbm - biaya_risiko_insiden
    
    # Estimasi waktu antri (Kasar, untuk laporan saja)
    # Total Jam Tersedia - (Siklus * Waktu Fisik Teoretis)
    waktu_fisik_per_siklus_jam = (31.76 + 11.02 + 25.29 + 8.10) / 60.0
    total_jam_kerja_alat = skenario['alokasi_truk'] * duration_hours
    total_jam_fisik = global_metrics['jumlah_siklus_selesai'] * waktu_fisik_per_siklus_jam
    total_jam_antri = max(0, total_jam_kerja_alat - total_jam_fisik)

    hasil_final = skenario.copy()
    hasil_final.update({
        'Z_SCORE_PROFIT': profit,
        'total_tonase': global_metrics['total_tonase'],
        'total_bbm_liter': global_metrics['total_bbm_liter'],
        'total_waktu_antri_jam': total_jam_antri, # Estimasi
        'jumlah_siklus_selesai': global_metrics['jumlah_siklus_selesai']
    })
    return hasil_final

# --- FUNGSI AGEN & FORMATTER ---

def format_currency(value):
    if value >= 1_000_000_000: return f"{value/1_000_000_000:.2f} Miliar IDR"
    elif value >= 1_000_000: return f"{value/1_000_000:.0f} Juta IDR"
    else: return f"{value:,.0f} IDR"

def get_operational_guidelines(weather, road_cond, total_trucks, total_excavators):
    guidelines = []
    if "Hujan" in str(weather):
        guidelines.append("⚠️ SLIPPERY: Batasi kecepatan maks 20 km/jam.")
    elif "Cerah" in str(weather):
        guidelines.append("✅ SPEED: Optimalkan kecepatan.")
    
    ratio = total_trucks / total_excavators
    if ratio > 5:
        guidelines.append("🔄 TRAFIK: Potensi antrian tinggi. Aktifkan waiting bay.")
    return guidelines

def format_konteks_for_llm(top_3_list):
    """
    Helper function: Menyiapkan Dashboard dengan DETAIL KONFIGURASI ALAT & RUTE.
    VERSI FINAL DASHBOARD READY: Struktur JSON sesuai dengan dashboard.py.
    """
    data_ringkas = []
    for i, res in enumerate(top_3_list, 1):
        
        # 1. Terjemahkan ID & Ambil Detail Tambahan
        road_id = res.get('target_road_id')
        excavator_id = res.get('target_excavator_id')
        
        road_name = "Unknown Road"
        road_dist = 0
        
        try:
            road_data = DB_ROADS.loc[road_id]
            road_name = road_data['name']
            road_dist = road_data['distance']
            road_info = f"{road_name} ({road_dist} km)"
        except:
            road_info = str(road_id)
            
        try:
            exc_data = DB_EXCAVATORS.loc[excavator_id]
            exc_name = exc_data['name']
            exc_loc = exc_data['currentLocation']
            excavator_info = f"{exc_loc} (Unit: {exc_name})"
        except:
            excavator_info = str(excavator_id)

        # 2. Format Angka
        # Pastikan fungsi format_currency ada di atas fungsi ini
        profit_fmt = format_currency(res.get('Z_SCORE_PROFIT', 0))
        tonase_fmt = f"{res.get('total_tonase', 0):,.0f} Ton"
        antri_fmt = f"{res.get('total_waktu_antri_jam', 0):.1f} Jam"
        
        fuel_ratio = 0
        if res.get('total_tonase', 0) > 0:
            fuel_ratio = res.get('total_bbm_liter', 0) / res.get('total_tonase', 1)
        fuel_ratio_fmt = f"{fuel_ratio:.2f} L/Ton"

        # 3. Ambil SOP
        sop_list = get_operational_guidelines(
            res.get('weatherCondition', 'Unknown'),
            res.get('roadCondition', 'Unknown'),
            res.get('alokasi_truk', 0),
            res.get('jumlah_excavator', 1)
        )
        sop_string = " | ".join(sop_list)

        # 4. Susun JSON dengan Struktur KPI_PREDIKSI (Wajib untuk Dashboard)
        data_ringkas.append({
            f"OPSI_{i}": {
                "TYPE": f"{'AGRESIF (Produksi Tinggi)' if i==1 else 'EFISIEN (Cost Rendah)'}",
                
                # Data untuk Tampilan Kartu
                "INSTRUKSI_FLAT": {
                    "JUMLAH_DUMP_TRUCK": f"{res.get('alokasi_truk')} Unit",
                    "JUMLAH_EXCAVATOR": f"{res.get('jumlah_excavator')} Unit",
                    "ALAT_MUAT_TARGET": excavator_info,
                    "JALUR_ANGKUT": road_info
                },
                
                # Data untuk Metrik Dashboard
                "KPI_PREDIKSI": {
                    "PROFIT": profit_fmt,
                    "PRODUKSI": tonase_fmt,
                    "FUEL_RATIO": fuel_ratio_fmt,
                    "IDLE_ANTRIAN": antri_fmt
                },
                
                "SOP_KESELAMATAN": sop_string
            }
        })
    return json.dumps(data_ringkas, indent=2, default=str)

def get_strategic_recommendations(fixed_conditions, decision_variables, financial_params):
    print(f"\n--- [Agen Strategis Hybrid] Mencari strategi terbaik... ---")
    keys, values = zip(*decision_variables.items())
    scenario_combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    skenario_list = []
    for combo in scenario_combinations:
        new_scenario = fixed_conditions.copy()
        new_scenario.update(combo)
        skenario_list.append(new_scenario)
        
    all_results = []
    for i, scenario in enumerate(skenario_list):
        print(f"  > Simulasi {i+1}/{len(skenario_list)}...")
        hasil = run_hybrid_simulation(scenario, financial_params, duration_hours=8)
        all_results.append(hasil)

    all_results.sort(key=lambda x: x['Z_SCORE_PROFIT'], reverse=True)
    return all_results[:3]

def run_follow_up_chat(top_3_strategies_list):
    if LLM_PROVIDER != "ollama":
        print("❌ ERROR: Server Ollama tidak terhubung.")
        return

    print(f"\n--- [Agen Chatbot ({OLLAMA_MODEL})] ---")
    data_context = format_konteks_for_llm(top_3_strategies_list)
    
    system_prompt = f"""
    PERAN: Kepala Teknik Tambang (KTT).
    DATA: {data_context}
    TUGAS: Analisis 3 strategi ini untuk Foreman.
    
    FORMAT JAWABAN (WAJIB):
    Gunakan pemisah "---BATAS_OPSI---" antar strategi.
    
    **STRATEGI 1: [JUDUL]**
    > Profit: [Angka]
    1. 📋 **KONFIGURASI**: [Detail Truk, Exc, Lokasi, Jalur]
    2. 📊 **ANALISIS**: [Produksi, Fuel Ratio, Antrian]
    3. ⚠️ **INSTRUKSI**: [Saran operasional & SOP]
    
    ---BATAS_OPSI---
    
    **STRATEGI 2: [JUDUL]**
    > Profit: [Angka]
    1. 📋 **KONFIGURASI**: [Detail Truk, Exc, Lokasi, Jalur]
    2. 📊 **ANALISIS**: [Produksi, Fuel Ratio, Antrian]
    3. ⚠️ **INSTRUKSI**: [Saran operasional & SOP]
    
    ---BATAS_OPSI---
    
    **STRATEGI 3: [JUDUL]**
    > Profit: [Angka]
    1. 📋 **KONFIGURASI**: [Detail Truk, Exc, Lokasi, Jalur]
    2. 📊 **ANALISIS**: [Produksi, Fuel Ratio, Antrian]
    3. ⚠️ **INSTRUKSI**: [Saran operasional & SOP]
    
    ---BATAS_OPSI---
    
    **KESIMPULAN**: [Rekomendasi final yang tegas]
    """
    
    chat_history = [{'role': 'system', 'content': system_prompt}]
    
    try:
        print("AI sedang menganalisis...")
        res = ollama.chat(model=OLLAMA_MODEL, messages=chat_history)
        print(f"\n{res['message']['content']}\n")
        chat_history.append(res['message'])
    except Exception as e:
        print(f"Error Ollama: {e}")
        return

    print("Ketik pertanyaan atau 'exit'.")
    while True:
        q = input("\n[Tanya]> ").strip()
        if q.lower() in ['exit', 'selesai']: break
        chat_history.append({'role': 'user', 'content': q})
        print("(Berpikir...)")
        try:
            res = ollama.chat(model=OLLAMA_MODEL, messages=chat_history)
            print(f"\n{res['message']['content']}\n")
            chat_history.append(res['message'])
        except Exception as e: 
             print(f"Error: {e}")
             break

# --- PEMUATAN GLOBAL ---
print("Memuat Sistem...")
CONFIG = load_config()
try:
    DB_TRUCKS = pd.read_csv(os.path.join(DATA_FOLDER, 'trucks.csv')).set_index('id')
    DB_EXCAVATORS = pd.read_csv(os.path.join(DATA_FOLDER, 'excavators.csv')).set_index('id')
    DB_OPERATORS = pd.read_csv(os.path.join(DATA_FOLDER, 'operators.csv')).set_index('id')
    DB_ROADS = pd.read_csv(os.path.join(DATA_FOLDER, 'road_segments.csv')).set_index('id')
    
    maint_logs = pd.read_csv(os.path.join(DATA_FOLDER, 'maintenance_logs.csv'))
    maint_logs = maint_logs[maint_logs['status'] == 'COMPLETED']
    maint_logs['completionDate'] = pd.to_datetime(maint_logs['completionDate'])
    DB_MAINTENANCE_SORTED = maint_logs.sort_values('completionDate')
    
    MODEL_FUEL = joblib.load(os.path.join(MODEL_FOLDER, 'model_fuel.joblib'))
    MODEL_LOAD = joblib.load(os.path.join(MODEL_FOLDER, 'model_load_weight.joblib'))
    MODEL_DELAY = joblib.load(os.path.join(MODEL_FOLDER, 'model_delay_probability.joblib'))
    
    with open(os.path.join(MODEL_FOLDER, 'numerical_columns.json')) as f: NUMERICAL_COLUMNS = json.load(f)
    with open(os.path.join(MODEL_FOLDER, 'categorical_columns.json')) as f: CATEGORICAL_COLUMNS = json.load(f)
    MODEL_COLUMNS = NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS
    
    import ollama
    ollama.list()
    LLM_PROVIDER, OLLAMA_MODEL = "ollama", "qwen2.5:7b"
    print("✅ Sistem Siap.")
except Exception as e:
    print(f"❌ Gagal Memuat: {e}")
    LLM_PROVIDER = None

# --- TEST BENCH ---
if __name__ == "__main__":
    if LLM_PROVIDER:
        fixed = {
            'weatherCondition': 'Hujan Ringan', 'roadCondition': 'FAIR', 'shift': 'SHIFT_1',
            'target_road_id': DB_ROADS.index[0], 'target_excavator_id': DB_EXCAVATORS.index[0]
        }
        vars = {'alokasi_truk': [5, 10], 'jumlah_excavator': [1, 2]}
        res = get_strategic_recommendations(fixed, vars, CONFIG['financial_params'])
        if res: run_follow_up_chat(res)