import pandas as pd
import joblib
import json
import warnings
import numpy as np
import ollama
import os
import simpy
from itertools import product

# --- 0. KONFIGURASI & PATH ---
warnings.filterwarnings('ignore')
DATA_FOLDER = 'data'
MODEL_FOLDER = 'models'

def load_config():
    """Memuat parameter finansial default."""
    return {
        'financial_params': {
            'HargaJualBatuBara': 800000, 
            'HargaSolar': 15000, 
            'BiayaPenaltiKeterlambatanKapal': 100000000,
            'BiayaRataRataInsiden': 50000000,
            'BiayaDemurragePerJam': 50000000
        }
    }

def format_currency(value):
    if value >= 1_000_000_000: return f"{value/1_000_000_000:.2f} Miliar IDR"
    elif value >= 1_000_000: return f"{value/1_000_000:.0f} Juta IDR"
    else: return f"{value:,.0f} IDR"

# --- 1. LOAD DATA GLOBAL ---
print("\n--- MEMUAT SISTEM (FINAL VERSION v3.0) ---")
CONFIG = load_config()

try:
    print(f"1. Memuat Database dari '{DATA_FOLDER}'...")
    DB_TRUCKS = pd.read_csv(os.path.join(DATA_FOLDER, 'trucks.csv')).set_index('id')
    DB_EXCAVATORS = pd.read_csv(os.path.join(DATA_FOLDER, 'excavators.csv')).set_index('id')
    DB_OPERATORS = pd.read_csv(os.path.join(DATA_FOLDER, 'operators.csv')).set_index('id')
    DB_ROADS = pd.read_csv(os.path.join(DATA_FOLDER, 'road_segments.csv')).set_index('id')
    
    # Muat Jadwal Kapal
    try:
        DB_SCHEDULES = pd.read_csv(os.path.join(DATA_FOLDER, 'sailing_schedules.csv')).set_index('id')
        
        # Gunakan errors='coerce' untuk mengubah data "0" atau format salah menjadi NaT (Null)
        # Gunakan format='mixed' agar bisa membaca berbagai format tanggal
        DB_SCHEDULES['etsLoading'] = pd.to_datetime(
            DB_SCHEDULES['etsLoading'], 
            errors='coerce', 
            format='mixed'
        )
        
        # Hapus jadwal yang tanggalnya rusak (NaT) agar tidak merusak perhitungan nanti
        invalid_count = DB_SCHEDULES['etsLoading'].isna().sum()
        if invalid_count > 0:
            print(f"   ⚠️ Warning: Ditemukan {invalid_count} jadwal dengan tanggal invalid (diabaikan).")
            DB_SCHEDULES = DB_SCHEDULES.dropna(subset=['etsLoading'])
            
        print(f"   > Schedules: {len(DB_SCHEDULES)} jadwal valid dimuat.")
        
        try:
            DB_VESSELS = pd.read_csv(os.path.join(DATA_FOLDER, 'vessels.csv')).set_index('id')
        except:
            DB_VESSELS = pd.DataFrame()
    except Exception as e:
        print(f"   ⚠️ Warning: Gagal memuat data kapal ({e})")
        DB_SCHEDULES = pd.DataFrame()

    # Muat Log Maintenance
    maint = pd.read_csv(os.path.join(DATA_FOLDER, 'maintenance_logs.csv'))
    maint = maint[maint['status'] == 'COMPLETED']
    maint['completionDate'] = pd.to_datetime(maint['completionDate'])
    DB_MAINTENANCE_SORTED = maint.sort_values('completionDate')
    
    print(f"2. Memuat Model ML dari '{MODEL_FOLDER}'...")
    MODEL_FUEL = joblib.load(os.path.join(MODEL_FOLDER, 'model_fuel.joblib'))
    MODEL_LOAD = joblib.load(os.path.join(MODEL_FOLDER, 'model_load_weight.joblib'))
    MODEL_DELAY = joblib.load(os.path.join(MODEL_FOLDER, 'model_delay_probability.joblib'))
    
    with open(os.path.join(MODEL_FOLDER, 'numerical_columns.json')) as f: NUM_COLS = json.load(f)
    with open(os.path.join(MODEL_FOLDER, 'categorical_columns.json')) as f: CAT_COLS = json.load(f)
    MODEL_COLUMNS = NUM_COLS + CAT_COLS
    
    print(f"3. Menghubungkan Ollama...")
    try:
        ollama.list()
        LLM_PROVIDER, OLLAMA_MODEL = "ollama", "llama3:8b" # Bisa diganti qwen2.5:7b
        print("✅ SISTEM SIAP.")
    except:
        print("⚠️ OLLAMA TIDAK TERHUBUNG.")
        LLM_PROVIDER = None
    
except Exception as e:
    print(f"❌ GAGAL MEMUAT SYSTEM: {e}")
    LLM_PROVIDER = None

# --- 2. LOGIKA SIMULASI ---

def get_features_for_prediction(truck_id, operator_id, road_id, excavator_id, weather, road_cond, shift, sim_time):
    """Membuat fitur untuk ML berdasarkan waktu simulasi (sim_time)"""
    try:
        truck = DB_TRUCKS.loc[truck_id]
        excavator = DB_EXCAVATORS.loc[excavator_id]
        operator = DB_OPERATORS.loc[operator_id]
        road = DB_ROADS.loc[road_id]
        
        # Hitung usia aset berdasarkan Waktu Simulasi
        truck_age_days = (sim_time - pd.to_datetime(truck['purchaseDate']).tz_convert(sim_time.tz)).days
        
        try: op_exp = json.loads(operator['competency']).get('years_experience', 0)
        except: op_exp = 0
            
        # Cari maintenance terakhir sebelum waktu simulasi
        last_maint = DB_MAINTENANCE_SORTED.loc[
            (DB_MAINTENANCE_SORTED['truckId'] == truck_id) &
            (DB_MAINTENANCE_SORTED['completionDate'] < sim_time)
        ]
        days_since_maint = 365 
        if not last_maint.empty:
            last_date = last_maint.iloc[-1]['completionDate'].tz_convert(sim_time.tz)
            days_since_maint = (sim_time - last_date).days

        feature_dict = {
            'capacity': truck['capacity'], 
            'bucketCapacity': excavator['bucketCapacity'],
            'rating': operator['rating'], 
            'operator_experience_years': op_exp,
            'distance': road['distance'], # Kunci 'distance' (Bukan distance_road)
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
    except Exception:
        return pd.DataFrame(columns=MODEL_COLUMNS)

def truck_process_hybrid(env, truck_id, operator_id, resources, global_metrics, skenario, sim_start_time):
    """
    Kehidupan satu truk.
    PERBAIKAN BUG: Logika hitung antrian (Queue Time) diperbaiki.
    """
    weather = skenario['weatherCondition']
    road_cond = skenario['roadCondition']
    shift = skenario['shift']
    
    excavator_id = skenario.get('target_excavator_id')
    if excavator_id not in DB_EXCAVATORS.index: excavator_id = DB_EXCAVATORS.index[0]
    road_id = skenario.get('target_road_id')
    if road_id not in DB_ROADS.index: road_id = DB_ROADS.index[0]

    excavator_resource = resources['excavator']
    
    try: kapasitas_ton = DB_TRUCKS.loc[truck_id]['capacity']
    except: return 

    weather_factor = 1.25 if "Hujan" in str(weather) else 1.0

    while True:
        # --- 1. PREDIKSI ML ---
        # Hitung waktu simulasi saat ini untuk fitur dinamis
        current_sim_time = sim_start_time + pd.Timedelta(seconds=env.now*3600)
        
        feats = get_features_for_prediction(truck_id, operator_id, road_id, excavator_id, weather, road_cond, shift, current_sim_time)
        
        fuel, load, delay = 10.0, kapasitas_ton * 0.87, 0.0
        if not feats.empty:
            try:
                fuel = MODEL_FUEL.predict(feats)[0]
                raw_load = MODEL_LOAD.predict(feats)[0]
                load = raw_load * 0.87
                delay = MODEL_DELAY.predict_proba(feats)[0][1]
            except: pass

        # --- 2. SIMULASI FISIK ---
        
        # A. HAULING (Pergi)
        avg_hauling = 31.76 * weather_factor
        yield env.timeout(avg_hauling / 60.0)
        
        # B. QUEUE & LOADING (Di sini perbaikan utamanya)
        # ---------------------------------------------------------
        waktu_masuk_antrian = env.now  # 1. Catat waktu datang
        
        with excavator_resource.request() as req:
            yield req  # 2. TUNGGU sampai dapat giliran (SimPy menghitung ini)
            
            # 3. Sudah dapat giliran! Catat waktu sekarang
            waktu_keluar_antrian = env.now
            
            # 4. Hitung durasi menunggu
            durasi_antri = waktu_keluar_antrian - waktu_masuk_antrian
            global_metrics['total_waktu_antri_jam'] += durasi_antri
            
            # 5. Proses Loading
            avg_loading = 11.02
            # Loading sedikit lebih lambat jika hujan
            actual_loading = avg_loading * (1.1 if "Hujan" in str(weather) else 1.0)
            yield env.timeout(actual_loading / 60.0)
        # ---------------------------------------------------------
            
        # C. RETURN (Pulang)
        avg_return = 25.29 * weather_factor
        yield env.timeout(avg_return / 60.0)
        
        # D. DUMPING
        avg_dump = 8.10
        yield env.timeout(avg_dump / 60.0)
        
        # --- 3. PENCATATAN ---
        global_metrics['total_tonase'] += load
        global_metrics['total_bbm_liter'] += (fuel * 1.6)
        global_metrics['total_probabilitas_delay'] += delay
        global_metrics['jumlah_siklus_selesai'] += 1

def calculate_shipment_risk(simulated_tonnage_8h, schedule_id, financial_params, sim_start_time):
    """
    Menghitung risiko demurrage DAN Estimasi Waktu Penyelesaian.
    """
    # ... (Bagian pengecekan ID dan load data jadwal SAMA SEPERTI SEBELUMNYA) ...
    if DB_SCHEDULES.empty or schedule_id not in DB_SCHEDULES.index:
        return {"status": "NO_SHIP", "demurrage_cost": 0, "info": "Tidak ada jadwal", "days_to_complete": 0}

    schedule = DB_SCHEDULES.loc[schedule_id]
    target = schedule.get('plannedQuantity', 0)
    current = schedule.get('actualQuantity', 0)
    if pd.isna(current): current = 0
    remaining_target = max(0, target - current)
    
    vessel_name = schedule['vesselId']
    if 'DB_VESSELS' in globals() and schedule['vesselId'] in DB_VESSELS.index:
        vessel_name = DB_VESSELS.loc[schedule['vesselId']]['name']

    # ... (Bagian deadline SAMA SEPERTI SEBELUMNYA) ...
    now = sim_start_time
    deadline = schedule['etsLoading']
    if deadline.tzinfo is None: deadline = deadline.tz_localize('UTC')
    time_remaining_hours = (deadline - now).total_seconds() / 3600.0
    
    if time_remaining_hours <= 0:
        return {
            "status": "LATE", "demurrage_cost": 100000000, 
            "info": f"Kapal {vessel_name} telat!", "vessel_name": vessel_name,
            "days_to_complete": 999 # Telat parah
        }

    # --- PERHITUNGAN BARU: ESTIMASI WAKTU SELESAI ---
    prod_per_hour = simulated_tonnage_8h / 8.0 # Kecepatan produksi per jam
    
    if prod_per_hour <= 0: 
        hours_needed = 9999
    else: 
        hours_needed = remaining_target / prod_per_hour

    days_needed = hours_needed / 24.0 # Konversi ke hari kerja (asumsi 24 jam kerja)
    # ------------------------------------------------

    variance = time_remaining_hours - hours_needed
    demurrage = 0
    status = "ON_SCHEDULE"
    
    if variance < 0:
        status = "DELAY_RISK"
        demurrage = abs(variance) * financial_params.get('BiayaDemurragePerJam', 50000000)

    return {
        "vessel_name": vessel_name, 
        "status": status, 
        "demurrage_cost": demurrage,
        "hours_needed": hours_needed,
        "days_to_complete": days_needed, # <-- DATA BARU
        "info": f"Selesai dalam {days_needed:.1f} Hari ({hours_needed:.0f} Jam)"
    }

def run_hybrid_simulation(skenario, financial_params, duration_hours=24):
    # Ambil waktu mulai simulasi (Default: Sekarang)
    sim_start_str = skenario.get('simulation_start_date', pd.Timestamp.now(tz='UTC').isoformat())
    try:
        sim_start_time = pd.to_datetime(sim_start_str)
        if sim_start_time.tzinfo is None: sim_start_time = sim_start_time.tz_localize('UTC')
    except:
        sim_start_time = pd.Timestamp.now(tz='UTC')

    env = simpy.Environment()
    res = {'excavator': simpy.Resource(env, capacity=skenario.get('jumlah_excavator', 1))}
    metrics = {
        'total_tonase': 0, 'total_bbm_liter': 0, 
        'jumlah_siklus_selesai': 0, 'total_waktu_antri_jam': 0.0,
        'total_probabilitas_delay': 0.0
    }
    
    trucks = DB_TRUCKS.index.tolist()
    ops = DB_OPERATORS.index.tolist()
    if not trucks: return skenario 

    for i in range(skenario['alokasi_truk']):
        t_id = trucks[i % len(trucks)]
        o_id = ops[i % len(ops)]
        # Pass sim_start_time ke proses truk
        env.process(truck_process_hybrid(env, t_id, o_id, res, metrics, skenario, sim_start_time))
        
    env.run(until=duration_hours)
    
    # Ekonomi
    p = financial_params
    rev = metrics['total_tonase'] * p['HargaJualBatuBara']
    cost = metrics['total_bbm_liter'] * p['HargaSolar']
    risk_antri = metrics['total_waktu_antri_jam'] * p['BiayaPenaltiKeterlambatanKapal']
    risk_insiden = metrics['total_probabilitas_delay'] * p.get('BiayaRataRataInsiden', 50000000)
    
    # Logistik Kapal
    schedule_id = skenario.get('target_schedule_id')
    ship_res = calculate_shipment_risk(metrics['total_tonase'], schedule_id, p, sim_start_time)
    biaya_demurrage = ship_res['demurrage_cost']
    
    profit = rev - cost - risk_antri - risk_insiden - biaya_demurrage
    
    result = skenario.copy()
    result.update(metrics)
    result.update({
        'Z_SCORE_PROFIT': profit,
        'total_biaya_risiko_antrian': risk_antri,
        'total_biaya_risiko_insiden': risk_insiden,
        'shipment_analysis': ship_res
    })
    return result

# --- 3. AGEN & HELPER ---

def get_operational_guidelines(weather, road_cond, total_trucks, total_excavators):
    guidelines = []
    if "Hujan" in str(weather): guidelines.append("⚠️ SLIPPERY: Batasi kecepatan 20 km/jam")
    if total_trucks/total_excavators > 5: guidelines.append("🔄 TRAFIK: Aktifkan Waiting Bay")
    return guidelines

def format_konteks_for_llm(top_3_list):
    data = []
    for i, res in enumerate(top_3_list, 1):
        r_id = res.get('target_road_id')
        e_id = res.get('target_excavator_id')
        
        # Helper Nama
        try: r_name = f"{DB_ROADS.loc[r_id]['name']} ({DB_ROADS.loc[r_id]['distance']} km)"
        except: r_name = str(r_id)
        try: e_name = f"{DB_EXCAVATORS.loc[e_id]['currentLocation']} (Unit: {DB_EXCAVATORS.loc[e_id]['name']})"
        except: e_name = str(e_id)

        # Format Angka
        profit_fmt = format_currency(res.get('Z_SCORE_PROFIT', 0))
        ton = res.get('total_tonase', 0)
        bbm = res.get('total_bbm_liter', 0)
        fr_fmt = f"{(bbm/ton):.2f} L/Ton" if ton>0 else "0"
        sop = get_operational_guidelines(res.get('weatherCondition'), res.get('roadCondition'), res.get('alokasi_truk'), res.get('jumlah_excavator'))
        
        # Data Kapal
        ship = res.get('shipment_analysis', {})
        ship_str = "Tidak ada kapal."
        estimasi_waktu = "N/A"
        
        if ship.get('status') != "NO_SHIP":
            days = ship.get('days_to_complete', 0)
            estimasi_waktu = f"{days:.1f} Hari" # Format: "2.5 Hari"
            
            ship_str = f"Kapal: {ship.get('vessel_name')} | Status: {ship.get('status')}"
            ship_str += f" | Estimasi Selesai: {estimasi_waktu}" # Info untuk AI
            
            if ship.get('demurrage_cost') > 0:
                ship_str += f" | DENDA: {format_currency(ship.get('demurrage_cost'))}"

        data.append({
            f"OPSI_{i}": {
                "TYPE": "REKOMENDASI UTAMA" if i==1 else "ALTERNATIF",
                "INSTRUKSI_FLAT": {
                    "JUMLAH_DUMP_TRUCK": f"{res.get('alokasi_truk')} Unit",
                    "JUMLAH_EXCAVATOR": f"{res.get('jumlah_excavator')} Unit",
                    "ALAT_MUAT_TARGET": e_name,
                    "JALUR_ANGKUT": r_name
                },
                "KPI_PREDIKSI": {
                    "PROFIT": profit_fmt,
                    "PRODUKSI": f"{ton:,.0f} Ton",
                    "ESTIMASI_DURASI": estimasi_waktu,
                    "FUEL_RATIO": fr_fmt,
                    "IDLE_ANTRIAN": f"{res.get('total_waktu_antri_jam', 0):.1f} Jam"
                },
                "ANALISIS_KAPAL": ship_str,
                "SOP_KESELAMATAN": " | ".join(sop)
            }
        })
    return json.dumps(data, indent=2)

def get_strategic_recommendations(fixed, vars, params):
    print(f"\n--- [Agen Strategis] Menjalankan Simulasi... ---")
    scenarios = [dict(zip(vars.keys(), v)) for v in product(*vars.values())]
    results = []
    for s in scenarios:
        full_s = fixed.copy()
        full_s.update(s)
        res = run_hybrid_simulation(full_s, params, duration_hours=8)
        results.append(res)
    results.sort(key=lambda x: x['Z_SCORE_PROFIT'], reverse=True)
    return results[:3]

def run_follow_up_chat(top_3):
    if not LLM_PROVIDER: return
    print("\n--- [Chatbot AI] Menganalisis... ---")
    context = format_konteks_for_llm(top_3)
    prompt = f"""
    PERAN: Kepala Teknik Tambang.
    DATA: {context}
    TUGAS: Analisis 3 strategi ini (Termasuk status Kapal/Demurrage).
    FORMAT WAJIB: Gunakan pemisah "---BATAS_OPSI---".
    """
    try:
        res = ollama.chat(model=OLLAMA_MODEL, messages=[{'role':'system', 'content':prompt}])
        print(f"\n{res['message']['content']}\n")
    except: pass

# --- 4. TEST BENCH ---
if __name__ == "__main__":
    if LLM_PROVIDER and not DB_ROADS.empty:
        sch_id = DB_SCHEDULES.index[0] if not DB_SCHEDULES.empty else None
        fixed = {
            'weatherCondition': 'Cerah', 'roadCondition': 'GOOD', 'shift': 'SHIFT_1',
            'target_road_id': DB_ROADS.index[0], 'target_excavator_id': DB_EXCAVATORS.index[0],
            'target_schedule_id': sch_id,
            'simulation_start_date': pd.Timestamp.now().isoformat() # Default Start Date
        }
        vars = {'alokasi_truk': [5, 10], 'jumlah_excavator': [1, 2]}
        res = get_strategic_recommendations(fixed, vars, CONFIG['financial_params'])
        if res: run_follow_up_chat(res)