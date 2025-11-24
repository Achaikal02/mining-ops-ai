import streamlit as st
import requests
import json
import pandas as pd
import os
from datetime import datetime

# --- KONFIGURASI ---
API_URL = "http://localhost:8000"
st.set_page_config(page_title="Mining Ops AI Command Center", layout="wide", page_icon="⛏️")

# --- STATE MANAGEMENT ---
if "strategies_context" not in st.session_state:
    st.session_state.strategies_context = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- CSS CUSTOM ---
st.markdown("""
<style>
    .stMetric {
        background-color: #000000;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #e9ecef;
    }
    .risk-alert {
        color: #721c24;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
        margin-top: 10px;
    }
    .safe-status {
        color: #155724;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
        margin-top: 10px;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #ddd;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- FUNGSI HELPER: MUAT DATA KAPAL (ANTI-GAGAL) ---
@st.cache_data
def load_vessel_options():
    """
    Membaca CSV Jadwal dan Kapal untuk membuat opsi dropdown.
    """
    try:
        sched_path = os.path.join('data', 'sailing_schedules.csv')
        ves_path = os.path.join('data', 'vessels.csv')
        
        if not os.path.exists(sched_path):
            return {"Data Jadwal Tidak Ditemukan": None}

        df_sched = pd.read_csv(sched_path)
        
        # Coba gabungkan dengan nama kapal
        has_name = False
        if os.path.exists(ves_path):
            try:
                df_vess = pd.read_csv(ves_path)
                df_sched = pd.merge(df_sched, df_vess, left_on='vesselId', right_on='id', suffixes=('', '_ves'))
                has_name = True
            except: pass

        # Filter Data Aktif
        active_df = df_sched
        if 'status' in df_sched.columns:
            active_df = df_sched[df_sched['status'].isin(['SCHEDULED', 'ARRIVED', 'LOADING', 'ANCHORAGE', 'STANDBY'])]
        
        if active_df.empty: active_df = df_sched.head(20)

        options = {}
        for _, row in active_df.iterrows():
            sch_id = row.get('id', 'UNKNOWN_ID')
            v_name = row.get('name', row.get('vesselId', 'Unknown Vessel'))
            
            eta_str = "N/A"
            if 'etaLoading' in row and pd.notna(row['etaLoading']):
                try: eta_str = pd.to_datetime(row['etaLoading']).strftime('%d-%b')
                except: pass
                
            target_str = "0"
            if 'plannedQuantity' in row and pd.notna(row['plannedQuantity']):
                target_str = f"{row['plannedQuantity']:,.0f}"
            
            label = f"{v_name} | ETA: {eta_str} | Target: {target_str} T"
            options[label] = sch_id
            
        return options
    except Exception:
        return {"Gagal memuat data kapal": None}

# --- TAB LAYOUT ---
tab_sim, tab_data = st.tabs(["🚀 Simulasi Operasional", "📝 Manajemen Data Kapal"])

# =============================================================================
# TAB 1: SIMULASI (DASHBOARD UTAMA)
# =============================================================================
with tab_sim:
    # --- SIDEBAR ---
    with st.sidebar:
        st.title("🎛️ Control Panel")
        
        st.markdown("### 📅 Waktu Simulasi")
        sim_date = st.date_input("Tanggal Mulai", value=datetime.now())
        sim_start_iso = f"{sim_date}T00:00:00Z" 
        
        with st.expander("1. Kondisi Lapangan", expanded=True):
            weather = st.selectbox("Cuaca", ["Cerah", "Hujan Ringan", "Hujan Lebat"], index=0)
            road = st.selectbox("Kondisi Jalan", ["GOOD", "FAIR", "POOR", "LICIN"], index=1)
            shift = st.selectbox("Shift", ["SHIFT_1", "SHIFT_2", "SHIFT_3"], index=0)
            
            # Dropdown Kapal Dinamis
            st.markdown("---")
            st.markdown("**🚢 Target Pengapalan**")
            vessel_map = load_vessel_options()
            selected_vessel_label = st.selectbox("Pilih Jadwal Kapal:", list(vessel_map.keys()))
            selected_schedule_id = vessel_map[selected_vessel_label]

        with st.expander("2. Variabel Keputusan", expanded=True):
            truck_opts = st.multiselect("Truk", [5, 10, 15, 20], default=[5, 10])
            exc_opts = st.multiselect("Excavator", [1, 2, 3], default=[1, 2])
            
        with st.expander("3. Parameter Ekonomi (What-If)", expanded=False):
            price_coal = st.number_input("Harga Batubara (IDR/Ton)", value=800000, step=10000)
            price_fuel = st.number_input("Harga Solar (IDR/Liter)", value=15000, step=500)
            cost_penalty = st.number_input("Denda Antri (IDR/Jam)", value=100000000, step=10000000)
            cost_incident = st.number_input("Biaya Insiden (IDR)", value=50000000, step=5000000)
            cost_demurrage = st.number_input("Denda Kapal (IDR/Jam)", value=50000000, step=5000000)
            
        st.divider()
        btn_simulate = st.button("🚀 JALANKAN SIMULASI", type="primary", use_container_width=True)

    # --- MAIN AREA ---
    st.title("⛏️ Mining Value Chain Optimizer")
    st.markdown("##### *Integrated Upstream-Downstream Decision Support System*")
    
    if btn_simulate:
        if not truck_opts or not exc_opts:
            st.error("Mohon pilih minimal satu opsi Truk dan Excavator.")
        else:
            with st.spinner(f"Menjalankan simulasi hybrid mulai {sim_date}..."):
                payload = {
                    "fixed_conditions": {
                        "weatherCondition": weather,
                        "roadCondition": road,
                        "shift": shift,
                        "target_road_id": "cmhsbjn8x02s2maft90hi31ty", # Default ID
                        "target_excavator_id": "cmhsbjpma05ddmaft5kv95dom",
                        "target_schedule_id": selected_schedule_id,
                        "simulation_start_date": sim_start_iso
                    },
                    "decision_variables": {
                        "alokasi_truk": truck_opts, "jumlah_excavator": exc_opts
                    },
                    "financial_params": {
                        "HargaJualBatuBara": price_coal,
                        "HargaSolar": price_fuel,
                        "BiayaPenaltiKeterlambatanKapal": cost_penalty,
                        "BiayaRataRataInsiden": cost_incident,
                        "BiayaDemurragePerJam": cost_demurrage
                    }
                }
                
                try:
                    response = requests.post(f"{API_URL}/get_top_3_strategies", json=payload)
                    if response.status_code == 200:
                        data = response.json()
                        st.session_state.strategies_context = data['top_3_strategies']
                        st.session_state.chat_history = []
                        st.toast("Simulasi Selesai!", icon="✅")
                    else:
                        st.error(f"API Error ({response.status_code}): {response.text}")
                except Exception as e:
                    st.error(f"Gagal terkoneksi ke Server API: {e}")

    # --- TAMPILAN HASIL ---
    if st.session_state.strategies_context:
        st.divider()
        cols = st.columns(3)
        
        for i, strat in enumerate(st.session_state.strategies_context):
            key_name = list(strat.keys())[0]
            details = strat[key_name]
            kpi = details['KPI_PREDIKSI']
            instr = details['INSTRUKSI_FLAT']
            kapal = details.get('ANALISIS_KAPAL', 'Tidak ada data kapal')
            
            with cols[i]:
                border_style = True 
                if i == 0: st.markdown("#### ⭐ REKOMENDASI UTAMA")
                elif i == 1: st.markdown("#### 🥈 OPSI EFISIEN")
                else: st.markdown("#### 🥉 ALTERNATIF")
                    
                with st.container(border=border_style):
                    st.metric("Estimasi Profit Shift", kpi['PROFIT'])
                    
                    c1, c2 = st.columns(2)
                    c1.metric("Produksi", kpi['PRODUKSI'])
                    c2.metric("Fuel Ratio", kpi['FUEL_RATIO'])
                    
                    st.metric("Idle/Antrian", kpi['IDLE_ANTRIAN'])
                    
                    st.markdown("---")
                    st.caption(f"🚢 **STATUS LOGISTIK**")
                    
                    if "LATE" in kapal or "DELAY_RISK" in kapal or "DENDA" in kapal:
                        st.markdown(f"<div class='risk-alert'>{kapal}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='safe-status'>{kapal}</div>", unsafe_allow_html=True)

                    st.markdown("---")
                    with st.expander("📋 Lihat Perintah Kerja"):
                        st.markdown(f"**ALAT:** {instr['JUMLAH_DUMP_TRUCK']} & {instr['JUMLAH_EXCAVATOR']}")
                        st.markdown(f"**LOKASI:** {instr['ALAT_MUAT_TARGET']}")
                        st.markdown(f"**JALUR:** {instr['JALUR_ANGKUT']}")
                        st.info(f"**SOP:** {details['SOP_KESELAMATAN']}")

        # --- CHATBOT ---
        st.divider()
        col_chat_L, col_chat_R = st.columns([1, 3])
        
        with col_chat_L:
            st.markdown("### 🤖 Asisten KTT")
            st.caption("AI Analyst (Ollama Local).")
            if st.button("Bersihkan Chat"):
                st.session_state.chat_history = []
                st.rerun()

        with col_chat_R:
            chat_container = st.container(height=400)
            with chat_container:
                for message in st.session_state.chat_history:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

            if prompt := st.chat_input("Ketik pertanyaan Anda di sini..."):
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                with chat_container:
                    with st.chat_message("user"):
                        st.markdown(prompt)

                with chat_container:
                    with st.chat_message("assistant"):
                        message_placeholder = st.empty()
                        message_placeholder.markdown("▌ *Sedang menganalisis...*")
                        
                        try:
                            payload_chat = {
                                "pertanyaan_user": prompt,
                                "top_3_strategies_context": st.session_state.strategies_context
                            }
                            res = requests.post(f"{API_URL}/ask_chatbot", json=payload_chat)
                            if res.status_code == 200:
                                ai_reply = res.json()['jawaban_ai']
                                message_placeholder.markdown(ai_reply)
                                st.session_state.chat_history.append({"role": "assistant", "content": ai_reply})
                            else:
                                message_placeholder.error("Gagal mendapatkan respon AI.")
                        except Exception as e:
                            message_placeholder.error(f"Error koneksi: {e}")
    else:
        st.info("👈 Silakan atur parameter di sidebar dan klik **'Jalankan Simulasi'**.")

# =============================================================================
# TAB 2: MANAJEMEN DATA
# =============================================================================
with tab_data:
    st.header("🚢 Manajemen Logistik Kapal")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Tambah Database Kapal")
        with st.form("form_vessel"):
            v_name = st.text_input("Nama Kapal", placeholder="MV. Coal Hunter")
            v_cap = st.number_input("Kapasitas (Ton)", value=50000)
            submit_v = st.form_submit_button("Simpan Kapal")
            
            if submit_v:
                try:
                    res = requests.post(f"{API_URL}/add_vessel", json={"name": v_name, "capacity": v_cap})
                    if res.status_code == 200: st.success("Kapal berhasil ditambahkan!")
                    else: st.error(f"Gagal: {res.text}")
                except: st.error("Koneksi API Gagal")

    with col2:
        st.subheader("2. Buat Jadwal Pengapalan Baru")
        try: 
            vessels_df = pd.read_csv('data/vessels.csv')
            vessel_list = vessels_df['name'].tolist()
            vessel_ids = vessels_df['id'].tolist()
            vessel_dict = dict(zip(vessel_list, vessel_ids))
        except: 
            vessel_list = ["Manual Input"]
            
        with st.form("form_schedule"):
            v_choice = st.selectbox("Pilih Kapal", vessel_list)
            s_buyer = st.text_input("Buyer", "PLN")
            s_qty = st.number_input("Target Muatan (Ton)", value=45000)
            
            c1, c2 = st.columns(2)
            s_eta = c1.date_input("ETA (Tiba)")
            s_etd = c2.date_input("ETS (Berangkat/Deadline)")
            
            submit_s = st.form_submit_button("Buat Jadwal")
            
            if submit_s:
                iso_eta = f"{s_eta}T00:00:00Z"
                iso_etd = f"{s_etd}T23:59:59Z"
                v_id = vessel_dict.get(v_choice, "unknown")
                
                payload_sched = {
                    "vesselId": v_id, "loadingPort": "Tarakan", "destination": "Surabaya",
                    "etaLoading": iso_eta, "etsLoading": iso_etd,
                    "plannedQuantity": s_qty, "buyer": s_buyer
                }
                try:
                    res = requests.post(f"{API_URL}/add_schedule", json=payload_sched)
                    if res.status_code == 200: st.success("Jadwal berhasil dibuat!")
                    else: st.error(f"Gagal: {res.text}")
                except: st.error("Koneksi API Gagal")
    
    st.divider()
    st.markdown("### 📋 Database Jadwal Saat Ini")
    try:
        df_show = pd.read_csv('data/sailing_schedules.csv')
        st.dataframe(df_show)
    except:
        st.info("Belum ada data jadwal.")