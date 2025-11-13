import pandas as pd
import joblib
import json
import warnings
import numpy as np
import ollama  # <--- PERUBAHAN
import os
import simpy
from itertools import product

# --- 0. Nonaktifkan Peringatan ---
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# --- 1. DEFINISI FUNGSI UTAMA (MESIN AI) ---

def load_config():
    """
    Memuat parameter finansial (hardcoded).
    """
    print("Memuat Konfigurasi Finansial (Hardcoded)...")
    default_params = {
        'HargaJualBatuBara': 800000, 
        'HargaSolar': 15000, 
        'BiayaPenaltiKeterlambatanKapal': 100000000
    }
    return {'financial_params': default_params}

def get_features_for_prediction(truck_id, operator_id, road_id, excavator_id, weather, road_cond, shift):
    """
    Mengumpulkan semua fitur untuk satu truk pada satu titik waktu
    dan mengembalikannya sebagai DataFrame 1-baris.
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
            'distance': road['distance'], # Menggunakan 'distance' yang sudah diperbaiki
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

    while True:
        # --- B. BUAT FITUR & PREDIKSI BBM (Hanya BBM) ---
        X_features = get_features_for_prediction(
            truck_id, operator_id, road_id, excavator_id, weather, road_cond, shift
        )
        
        fuel_consumed_pred = 10.0 # Default
        if not X_features.empty:
            try:
                fuel_consumed_pred = MODEL_FUEL.predict(X_features)[0]
            except Exception:
                pass # Gunakan default
        
        # --- C. SIMULASI PROSES FISIK (SEMUA DIKALIBRASI) ---
        avg_hauling_menit = 31.76
        hauling_duration_jam = avg_hauling_menit / 60.0
        yield env.timeout(hauling_duration_jam)
        
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
        
        # --- D. CATAT HASIL SIKLUS (Dengan Perbaikan BBM * 2) ---
        global_metrics['total_tonase'] += kapasitas_ton
        global_metrics['total_bbm_liter'] += (fuel_consumed_pred * 2) 
        global_metrics['jumlah_siklus_selesai'] += 1

def run_hybrid_simulation(skenario, duration_hours=24):
    """
    Fungsi "MESIN" Hybrid.
    """
    env = simpy.Environment()
    
    jumlah_excavator_aktif = skenario.get('jumlah_excavator', 1)
    resources = {'excavator': simpy.Resource(env, capacity=jumlah_excavator_aktif)}
    
    global_metrics = {
        'total_tonase': 0, 'total_bbm_liter': 0, 
        'jumlah_siklus_selesai': 0, 'total_waktu_antri_jam': 0.0
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
    
    harga_jual_ton = CONFIG['financial_params']['HargaJualBatuBara']
    harga_solar_liter = CONFIG['financial_params']['HargaSolar']
    biaya_penalti_delay = CONFIG['financial_params']['BiayaPenaltiKeterlambatanKapal']

    pendapatan = global_metrics['total_tonase'] * harga_jual_ton
    biaya_bbm = global_metrics['total_bbm_liter'] * harga_solar_liter
    biaya_risiko = global_metrics['total_waktu_antri_jam'] * biaya_penalti_delay
    profit = pendapatan - biaya_bbm - biaya_risiko
    
    hasil_final = skenario.copy()
    hasil_final.update({
        'Z_SCORE_PROFIT': profit,
        'total_tonase': global_metrics['total_tonase'],
        'total_bbm_liter': global_metrics['total_bbm_liter'],
        'total_waktu_antri_jam': global_metrics['total_waktu_antri_jam'],
        'jumlah_siklus_selesai': global_metrics['jumlah_siklus_selesai']
    })
    return hasil_final

def get_strategic_recommendations(fixed_conditions, decision_variables):
    """
    AGEN 1: STRATEGIC OPTIMIZER (Si "Profesor")
    """
    print(f"\n--- [Agen Strategis Hybrid] Mencari 3 strategi terbaik untuk: {fixed_conditions} ---")
    
    keys, values = zip(*decision_variables.items())
    scenario_combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    skenario_list = []
    for combo in scenario_combinations:
        new_scenario = fixed_conditions.copy()
        new_scenario.update(combo)
        skenario_list.append(new_scenario)
        
    print(f"Total {len(skenario_list)} skenario strategi akan disimulasikan (ini mungkin butuh waktu)...")

    all_results = []
    for i, scenario in enumerate(skenario_list):
        print(f"  Menjalankan simulasi hybrid {i+1}/{len(skenario_list)}...")
        hasil = run_hybrid_simulation(scenario, duration_hours=8)
        all_results.append(hasil)

    all_results.sort(key=lambda x: x['Z_SCORE_PROFIT'], reverse=True)
    top_3_list = all_results[:3]
    print(f"✅ Analisis Strategis Hybrid Selesai. 3 strategi terbaik ditemukan.")
    return top_3_list

def format_konteks_for_llm(top_3_list):
    """
    Helper function untuk memformat data Top 3 untuk LLM.
    """
    data_ringkas = []
    for i, res in enumerate(top_3_list, 1):
        data_ringkas.append({
            f"STRATEGI_{i}": {
                "alokasi_truk": res.get('alokasi_truk'),
                "jumlah_excavator": res.get('jumlah_excavator'),
                "target_road_id": res.get('target_road_id'),
                "Z_SCORE_PROFIT": res.get('Z_SCORE_PROFIT'),
                "total_tonase": res.get('total_tonase'),
                "total_bbm_liter": res.get('total_bbm_liter'),
                "total_waktu_antri_jam": res.get('total_waktu_antri_jam')
            }
        })
    return json.dumps(data_ringkas, indent=2, default=str)

def run_follow_up_chat(top_3_strategies_list):
    """
    AGEN 2: FOLLOW-UP CHATBOT (Ditenagai OLLAMA)
    """
    if LLM_PROVIDER != "ollama":
        print("❌ ERROR: Server Ollama tidak terhubung. Chatbot tidak bisa dimulai.")
        return

    print(f"\n--- [Agen Chatbot v7.1 - Ollama ({OLLAMA_MODEL})] ---")
    print("AI sedang menganalisis 3 strategi terbaik untuk Anda...")

    data_konteks_string = format_konteks_for_llm(top_3_strategies_list)
    
    # --- PROMPT SISTEM UNTUK OLLAMA ---
    system_prompt = f"""
    PERAN ANDA:
    Anda adalah Asisten Analis Operasi Tambang yang cerdas dan profesional.
    Tugas Anda adalah membantu operator memahami 3 STRATEGI TERBAIK yang telah dipilihkan.
    Gunakan Bahasa Indonesia yang jelas dan profesional.

    DATA 3 STRATEGI TERBAIK (JSON):
    {data_konteks_string}
    
    ATURAN:
    1. Jawab pertanyaan HANYA berdasarkan 3 strategi dalam DATA KONTEKS di atas.
    2. Fokus Anda HANYA pada 3 strategi ini.
    3. Gunakan 'Z_SCORE_PROFIT' sebagai metrik utama.
    4. Jelaskan pro dan kontra (misal: "Strategi 1 profitnya tertinggi, tapi waktu antrinya juga tinggi...").
    5. JANGAN mengarang jawaban di luar data.
    """
    
    # --- PROMPT TUGAS PEMBUKA (BARU) ---
    task_prompt = """
    Tolong rangkum 3 strategi ini untuk saya. 
    Beri judul yang jelas untuk setiap strategi (REKOMENDASI UTAMA, PILIHAN EFISIEN, dll). 
    Jelaskan Pro (Keuntungan) dan Kontra (Risiko/Biaya) untuk masing-masing, 
    fokus pada Profit, Tonase, dan Waktu Antri.
    """
    
    # Buat riwayat chat
    chat_history = [
        {'role': 'system', 'content': system_prompt}
        # Kita akan tambahkan pesan pertama secara dinamis
    ]
    
    try:
        print("[Agen Chatbot]> (Sedang menganalisis...)")
        
        # Tambahkan tugas pembuka
        chat_history.append({'role': 'user', 'content': task_prompt})
        
        # --- PANGGIL OLLAMA ---
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=chat_history
        )
        
        jawaban_pembuka = response['message']['content']
        
        # Tambahkan jawaban AI ke riwayat
        chat_history.append({'role': 'assistant', 'content': jawaban_pembuka})
        
        print("\n--- ANALISIS STRATEGIS DARI AI ---")
        print(jawaban_pembuka) # Tampilkan analisis pembuka
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
CONFIG = load_config()

print("Memuat Agen Prediksi (ML) dan Database CSV...")
try:
    DB_TRUCKS = pd.read_csv('trucks.csv').set_index('id')
    DB_EXCAVATORS = pd.read_csv('excavators.csv').set_index('id')
    DB_OPERATORS = pd.read_csv('operators.csv').set_index('id')
    DB_ROADS = pd.read_csv('road_segments.csv').set_index('id')
    
    maint_logs_clean = pd.read_csv('maintenance_logs.csv')
    maint_logs_clean = maint_logs_clean.loc[maint_logs_clean['status'] == 'COMPLETED'].copy()
    maint_logs_clean['completionDate'] = pd.to_datetime(maint_logs_clean['completionDate'])
    DB_MAINTENANCE_SORTED = maint_logs_clean[['truckId', 'completionDate']].sort_values('completionDate')
    print("✅ Database CSV berhasil dimuat.")
    
    MODEL_HAULING = joblib.load('model_hauling_duration.joblib')
    MODEL_FUEL = joblib.load('model_fuel.joblib')
    print("✅ Model ML (.joblib) berhasil dimuat.")
    
    with open('numerical_columns.json', 'r') as f:
        NUMERICAL_COLUMNS = json.load(f)
    with open('categorical_columns.json', 'r') as f:
        CATEGORICAL_COLUMNS = json.load(f)
    MODEL_COLUMNS = NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS
    print("✅ Konfigurasi kolom model (.json) berhasil dimuat.")
    
except FileNotFoundError as e:
    print(f"❌ ERROR: File data penting ({e.filename}) tidak ditemukan.")
    MODEL_HAULING = None
except Exception as e:
    print(f"Error saat memuat model/data: {e}")
    MODEL_HAULING = None

print("Memuat Agen Q&A (Ollama)...")
try:
    # --- PERUBAHAN DI SINI ---
    import ollama
    ollama.list() 
    print("✅ Agen Q&A (Ollama) berhasil terhubung.")
    LLM_PROVIDER = "ollama"
    OLLAMA_MODEL = 'llama3:8b' # Ganti ini jika perlu
except Exception as e:
    print(f"PERINGATAN: Gagal terhubung ke server Ollama (pastikan sudah berjalan di localhost:11434).")
    print(f"Error: {e}")
    LLM_PROVIDER = None
# ------------------------------

# --- 3. TEST BENCH (Menjalankan Arsitektur Final) ---
if __name__ == "__main__":
    
    if CONFIG and MODEL_HAULING and LLM_PROVIDER == "ollama":
        
        print("\n--- [Test Bench Arsitektur Hybrid (SimPy + ML + Ollama)] ---")
        
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

        top_3_strategies = get_strategic_recommendations(fixed_conditions, decision_variables)
        
        if top_3_strategies:
            print("\n--- [HASIL REKOMENDASI STRATEGIS (TOP 3)] ---")
            print("3 strategi terbaik telah ditemukan. Menyerahkan ke Agen Chatbot Ollama...")
            print("---------------------------------")
            
            run_follow_up_chat(top_3_strategies)
        else:
            print("Tidak dapat menghasilkan rekomendasi strategi.")
    else:
        print("Gagal memuat Konfigurasi, Model ML, atau terhubung ke Ollama. Test bench dibatalkan.")