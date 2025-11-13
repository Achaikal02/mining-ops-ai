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

# --- 1. DEFINISI FUNGSI UTAMA (MESIN AI) ---

def load_config():
    """
    Memuat parameter finansial (hardcoded).
    VERSI UPGRADE V2.0: Menambahkan Biaya Insiden
    """
    print("Memuat Konfigurasi Finansial (Hardcoded)...")
    default_params = {
        'HargaJualBatuBara': 800000, 
        'HargaSolar': 15000, 
        'BiayaPenaltiKeterlambatanKapal': 100000000, # Biaya per JAM ANTRI
        'BiayaRataRataInsiden': 50000000           # Biaya per INSIDEN (ban pecah, dll)
    }
    return {'financial_params': default_params}

def get_features_for_prediction(truck_id, operator_id, road_id, excavator_id, weather, road_cond, shift):
    """
    (Fungsi ini tidak berubah, tugasnya masih sama)
    """
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
            'distance': road['distance'], 
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
        print(f"Peringatan: Gagal menemukan data untuk {e}.")
        return pd.DataFrame(columns=MODEL_COLUMNS)
    except Exception as e:
        print(f"Error saat membuat fitur: {e}")
        return pd.DataFrame(columns=MODEL_COLUMNS)

def truck_process_hybrid(env, truck_id, operator_id, resources, global_metrics, skenario):
    """
    'Kehidupan' satu truk (SimPy + ML).
    VERSI UPGRADE V2.0: Sekarang memprediksi BBM, MUATAN, dan RISIKO INSIDEN.
    """
    
    weather = skenario['weatherCondition']
    road_cond = skenario['roadCondition']
    shift = skenario['shift']
    excavator_resource = resources['excavator']
    excavator_id = skenario.get('target_excavator_id')
    road_id = skenario.get('target_road_id')
    
    # Kapasitas default jika truk tidak ditemukan
    kapasitas_ton = 30 
    try:
        truck_data = DB_TRUCKS.loc[truck_id]
        kapasitas_ton = truck_data['capacity'] # Ambil kapasitas maks
    except KeyError:
        print(f"Truk {truck_id} tidak ditemukan.")

    while True:
        # --- B. BUAT FITUR & PREDIKSI (BBM, MUATAN, & DELAY) ---
        X_features = get_features_for_prediction(
            truck_id, operator_id, road_id, excavator_id, weather, road_cond, shift
        )
        
        fuel_consumed_pred = 10.0 # Default
        load_weight_pred = kapasitas_ton # Default
        delay_proba_pred = 0.0 # Default
        
        if not X_features.empty:
            try:
                fuel_consumed_pred = MODEL_FUEL.predict(X_features)[0]
                load_weight_pred = MODEL_LOAD.predict(X_features)[0]
                delay_proba_pred = MODEL_DELAY.predict_proba(X_features)[0][1] # [0]=False, [1]=True
            except Exception as e:
                print(f"ERROR prediksi ML: {e}. Menggunakan default.")
        
        
        # --- C. SIMULASI PROSES FISIK (KALIBRASI) ---
        avg_hauling_menit = 31.76
        yield env.timeout(avg_hauling_menit / 60.0)
        
        waktu_masuk_antrian = env.now
        with excavator_resource.request() as req:
            yield req 
            waktu_keluar_antrian = env.now
            global_metrics['total_waktu_antri_jam'] += (waktu_keluar_antrian - waktu_masuk_antrian)
            avg_loading_menit = 11.02
            yield env.timeout(avg_loading_menit / 60.0)
        
        avg_return_menit = 25.29
        yield env.timeout(avg_return_menit / 60.0)
        
        avg_dumping_menit = 8.10
        yield env.timeout(avg_dumping_menit / 60.0)
        
        # --- D. CATAT HASIL SIKLUS ---
        global_metrics['total_tonase'] += load_weight_pred # <-- Gunakan Tonase Prediksi
        global_metrics['total_bbm_liter'] += (fuel_consumed_pred * 2) 
        global_metrics['total_probabilitas_delay'] += delay_proba_pred
        global_metrics['jumlah_siklus_selesai'] += 1

def run_hybrid_simulation(skenario, financial_params, duration_hours=24):
    """
    Fungsi "MESIN" Hybrid.
    VERSI UPGRADE V2.0: Menghitung 2 jenis risiko.
    """
    env = simpy.Environment()
    
    jumlah_excavator_aktif = skenario.get('jumlah_excavator', 1)
    resources = {'excavator': simpy.Resource(env, capacity=jumlah_excavator_aktif)}
    
    global_metrics = {
        'total_tonase': 0, 'total_bbm_liter': 0, 
        'jumlah_siklus_selesai': 0, 'total_waktu_antri_jam': 0.0,
        'total_probabilitas_delay': 0.0 # <-- Metrik baru
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
    
    # --- Hitung Hasil Akhir (Profit) ---
    harga_jual_ton = financial_params['HargaJualBatuBara']
    harga_solar_liter = financial_params['HargaSolar']
    biaya_penalti_antri = financial_params['BiayaPenaltiKeterlambatanKapal']
    biaya_per_insiden = financial_params['BiayaRataRataInsiden']

    pendapatan = global_metrics['total_tonase'] * harga_jual_ton
    biaya_bbm = global_metrics['total_bbm_liter'] * harga_solar_liter
    
    # --- PERHITUNGAN RISIKO BARU ---
    biaya_risiko_antrian = global_metrics['total_waktu_antri_jam'] * biaya_penalti_antri
    biaya_risiko_insiden = global_metrics['total_probabilitas_delay'] * biaya_per_insiden
    # -----------------------------
    
    profit = pendapatan - biaya_bbm - biaya_risiko_antrian - biaya_risiko_insiden
    
    hasil_final = skenario.copy()
    hasil_final.update({
        'Z_SCORE_PROFIT': profit,
        'total_tonase': global_metrics['total_tonase'],
        'total_bbm_liter': global_metrics['total_bbm_liter'],
        'total_waktu_antri_jam': global_metrics['total_waktu_antri_jam'],
        'total_biaya_risiko_antrian': biaya_risiko_antrian,
        'total_biaya_risiko_insiden': biaya_risiko_insiden,
        'jumlah_siklus_selesai': global_metrics['jumlah_siklus_selesai']
    })
    return hasil_final

# --- 2. DEFINISI DUA AGEN (APLIKASI AI) ---

def get_strategic_recommendations(fixed_conditions, decision_variables, financial_params):
    """
    (Fungsi ini tidak berubah, ia hanya memanggil 'run_hybrid_simulation' V2.0)
    """
    print(f"\n--- [Agen Strategis Hybrid V2.0] Mencari 3 strategi terbaik untuk: {fixed_conditions} ---")
    
    keys, values = zip(*decision_variables.items())
    scenario_combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    skenario_list = []
    for combo in scenario_combinations:
        new_scenario = fixed_conditions.copy()
        new_scenario.update(combo)
        skenario_list.append(new_scenario)
        
    print(f"Total {len(skenario_list)} skenario strategi akan disimulasikan...")

    all_results = []
    for i, scenario in enumerate(skenario_list):
        print(f"  Menjalankan simulasi hybrid V2.0 {i+1}/{len(skenario_list)}...")
        hasil = run_hybrid_simulation(scenario, financial_params, duration_hours=8)
        all_results.append(hasil)

    all_results.sort(key=lambda x: x['Z_SCORE_PROFIT'], reverse=True)
    top_3_list = all_results[:3]
    print(f"✅ Analisis Strategis Hybrid Selesai. 3 strategi terbaik ditemukan.")
    return top_3_list

def format_konteks_for_llm(top_3_list):
    """
    Helper function untuk memformat data Top 3 untuk LLM.
    VERSI BARU: Menambahkan 2 jenis biaya risiko.
    """
    data_ringkas = []
    for i, res in enumerate(top_3_list, 1):
        
        road_id = res.get('target_road_id')
        excavator_id = res.get('target_excavator_id')
        try:
            road_name = DB_ROADS.loc[road_id]['name']
        except KeyError:
            road_name = road_id
        try:
            excavator_name = DB_EXCAVATORS.loc[excavator_id]['name']
        except KeyError:
            excavator_name = excavator_id

        data_ringkas.append({
            f"STRATEGI_{i}": {
                "alokasi_truk": res.get('alokasi_truk'),
                "jumlah_excavator": res.get('jumlah_excavator'),
                "target_rute": road_name,       
                "target_excavator": excavator_name, 
                "ESTIMASI_PROFIT_SHIFT": res.get('Z_SCORE_PROFIT'),
                "total_tonase": res.get('total_tonase'),
                "total_bbm_liter": res.get('total_bbm_liter'),
                "total_waktu_antri_jam": res.get('total_waktu_antri_jam'),
                # --- TAMBAHKAN DATA RISIKO BARU ---
                "biaya_risiko_antrian": res.get('total_biaya_risiko_antrian'),
                "biaya_risiko_insiden": res.get('total_biaya_risiko_insiden')
            }
        })
    return json.dumps(data_ringkas, indent=2, default=str)

def run_follow_up_chat(top_3_strategies_list):
    """
    AGEN 2: FOLLOW-UP CHATBOT (OLLAMA)
    VERSI BARU: Promptnya sekarang tahu tentang 2 jenis risiko.
    """
    if LLM_PROVIDER != "ollama":
        print("❌ ERROR: Server Ollama tidak terhubung. Chatbot tidak bisa dimulai.")
        return

    print(f"\n--- [Agen Chatbot v7.3 - Ollama ({OLLAMA_MODEL})] ---")
    print("AI sedang menganalisis 3 strategi terbaik untuk Anda...")

    data_konteks_string = format_konteks_for_llm(top_3_strategies_list)
    
    # --- PROMPT SISTEM BARU ---
    analisis_pembuka = f"""
    !!! PERINTAH UTAMA: RESPONS ANDA HARUS SELALU DALAM BAHASA INDONESIA. !!!

    PERAN ANDA:
    Anda adalah Asisten Analis Operasi Tambang.
    Tugas Anda adalah menganalisis 3 STRATEGI TERBAIK yang diberikan.
    Data ini memiliki 2 JENIS BIAYA RISIKO:
    1. 'biaya_risiko_antrian': Biaya kerugian akibat truk mengantri (bottleneck).
    2. 'biaya_risiko_insiden': Biaya kerugian akibat prediksi insiden (ban pecah, dll).

    DATA 3 STRATEGI TERBAIK (JSON):
    {data_konteks_string}
    
    TUGAS ANDA (DALAM BAHASA INDONESIA):
    1.  Tulis rangkuman 3 strategi. Beri judul (REKOMENDASI UTAMA, dll).
    2.  Untuk setiap strategi, jelaskan Pro & Kontra.
    3.  Fokus pada 'ESTIMASI_PROFIT_SHIFT', 'total_tonase', 'biaya_risiko_antrian', dan 'biaya_risiko_insiden'.
    4.  Akhiri dengan "Silakan ajukan pertanyaan jika ada detail yang ingin Anda ketahui."
    """
    
    chat_history = [{'role': 'system', 'content': analisis_pembuka}]
    
    try:
        print("[Agen Chatbot]> (Sedang menganalisis...)")
        
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=chat_history 
        )
        
        jawaban_pembuka = response['message']['content']
        chat_history.append({'role': 'assistant', 'content': jawaban_pembuka})
        
        print("\n--- ANALISIS STRATEGIS DARI AI ---")
        print(jawaban_pembuka) 
        print("\n---------------------------------")
        print("Ketik 'selesai' atau 'exit' untuk keluar.")

    except Exception as e:
         print(f"[Agen Chatbot]> Terjadi error saat menghasilkan analisis pembuka: {e}")
         return

    # Loop untuk pertanyaan lanjutan
    while True:
        try:
            pertanyaan = input("\n[User Q&A Lanjutan]> ").strip()
            if pertanyaan.lower() in ['selesai', 'exit']:
                print("[Agen Chatbot]> Sesi Q&A ditutup. Semoga berhasil.")
                break
            if not pertanyaan: continue
            
            chat_history.append({'role': 'user', 'content': pertanyaan})
            print("[Agen Chatbot]> (Sedang berpikir...)")
            
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=chat_history
            )
            
            jawaban_ai = response['message']['content']
            chat_history.append({'role': 'assistant', 'content': jawaban_ai})
            
            print(f"\n[Agen Chatbot]> {jawaban_ai}")
        
        except Exception as e:
             print(f"[Agen Chatbot]> Terjadi error saat menghubungi Ollama: {e}")
             break

# --- 2. PEMUATAN GLOBAL (Hanya berjalan 1x saat script di-load) ---

print("Memuat Konfigurasi Awal...")
CONFIG = load_config() # <-- SEKARANG BERISI BIAYA INSIDEN

print("Memuat Agen Prediksi (ML) dan Database CSV...")
try:
    print(f"Memuat database CSV mentah dari '{DATA_FOLDER}'...")
    DB_TRUCKS = pd.read_csv(os.path.join(DATA_FOLDER, 'trucks.csv')).set_index('id')
    DB_EXCAVATORS = pd.read_csv(os.path.join(DATA_FOLDER, 'excavators.csv')).set_index('id')
    DB_OPERATORS = pd.read_csv(os.path.join(DATA_FOLDER, 'operators.csv')).set_index('id')
    DB_ROADS = pd.read_csv(os.path.join(DATA_FOLDER, 'road_segments.csv')).set_index('id')
    
    maint_logs_clean = pd.read_csv(os.path.join(DATA_FOLDER, 'maintenance_logs.csv'))
    maint_logs_clean = maint_logs_clean.loc[maint_logs_clean['status'] == 'COMPLETED'].copy()
    maint_logs_clean['completionDate'] = pd.to_datetime(maint_logs_clean['completionDate'])
    DB_MAINTENANCE_SORTED = maint_logs_clean[['truckId', 'completionDate']].sort_values('completionDate')
    print("✅ Database CSV berhasil dimuat.")
    
    # --- Muat Model ML dari /models (SEKARANG 3 MODEL) ---
    print(f"Memuat 3 model ML (.joblib) dari '{MODEL_FOLDER}'...")
    MODEL_FUEL = joblib.load(os.path.join(MODEL_FOLDER, 'model_fuel.joblib'))
    MODEL_LOAD = joblib.load(os.path.join(MODEL_FOLDER, 'model_load_weight.joblib'))
    MODEL_DELAY = joblib.load(os.path.join(MODEL_FOLDER, 'model_delay_probability.joblib'))
    print("✅ 3 Model ML (Fuel, Load, Delay) berhasil dimuat.")
    
    # --- Muat Konfigurasi Kolom dari /models ---
    with open(os.path.join(MODEL_FOLDER, 'numerical_columns.json'), 'r') as f:
        NUMERICAL_COLUMNS = json.load(f)
    with open(os.path.join(MODEL_FOLDER, 'categorical_columns.json'), 'r') as f:
        CATEGORICAL_COLUMNS = json.load(f)
    MODEL_COLUMNS = NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS
    print("✅ Konfigurasi kolom model (.json) berhasil dimuat.")
    
except FileNotFoundError as e:
    print(f"❌ ERROR: File data penting ({e.filename}) tidak ditemukan.")
    MODEL_FUEL = None 
except Exception as e:
    print(f"Error saat memuat model/data: {e}")
    MODEL_FUEL = None

print("Memuat Agen Q&A (Ollama)...")
try:
    import ollama
    ollama.list() 
    print("✅ Agen Q&A (Ollama) berhasil terhubung.")
    LLM_PROVIDER = "ollama"
    OLLAMA_MODEL = 'llama3:8b'
except Exception as e:
    print(f"PERINGATAN: Gagal terhubung ke server Ollama.")
    LLM_PROVIDER = None

# --- 3. TEST BENCH (Menjalankan Arsitektur Final) ---
if __name__ == "__main__":
    
    # Perbarui cek: Sekarang kita cek MODEL_FUEL
    if CONFIG and MODEL_FUEL and LLM_PROVIDER == "ollama":
        
        print("\n--- [Test Bench Arsitektur Hybrid (V2.0: 3 Model)] ---")
        
        fixed_conditions = {
            'weatherCondition': 'Hujan Ringan',
            'roadCondition': 'FAIR',
            'shift': 'SHIFT_1',
            'target_road_id': DB_ROADS.index[0],
            'target_excavator_id': DB_EXCAVATORS.index[0] 
        }
        
        decision_variables = {
            'alokasi_truk': [5, 10],            
            'jumlah_excavator': [1, 2]         
        }

        top_3_strategies = get_strategic_recommendations(
            fixed_conditions, 
            decision_variables, 
            CONFIG['financial_params']
        )
        
        if top_3_strategies:
            run_follow_up_chat(top_3_strategies)
        else:
            print("Tidak dapat menghasilkan rekomendasi strategi.")
    else:
        print("Gagal memuat Konfigurasi, Model ML, atau terhubung ke Ollama. Test bench dibatalkan.")