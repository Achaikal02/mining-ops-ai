import streamlit as st
import requests
import json

# --- KONFIGURASI ---
API_URL = "http://localhost:8000"
st.set_page_config(page_title="Mining Ops AI Assistant", layout="wide")

# --- STATE MANAGEMENT ---
# Menyimpan konteks strategi agar tidak hilang saat chat
if "strategies_context" not in st.session_state:
    st.session_state.strategies_context = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- SIDEBAR: INPUT USER ---
with st.sidebar:
    st.header("⚙️ Parameter Operasional")
    
    st.subheader("1. Kondisi Lapangan")
    weather = st.selectbox("Cuaca", ["Cerah", "Hujan Ringan", "Hujan Lebat"])
    road = st.selectbox("Kondisi Jalan", ["GOOD", "FAIR", "POOR", "LICIN"])
    shift = st.selectbox("Shift", ["SHIFT_1", "SHIFT_2", "SHIFT_3"])
    
    st.subheader("2. Variabel Keputusan")
    # Multiselect untuk opsi
    truck_opts = st.multiselect("Opsi Jumlah Truk", [5, 10, 15, 20], default=[5, 10])
    exc_opts = st.multiselect("Opsi Jumlah Excavator", [1, 2, 3], default=[1, 2])
    
    with st.expander("💰 Parameter Finansial (What-If)"):
        price_coal = st.number_input("Harga Jual (IDR/Ton)", value=800000)
        price_fuel = st.number_input("Harga Solar (IDR/Liter)", value=15000)
        cost_penalty = st.number_input("Denda Antri (IDR/Jam)", value=100000000)

    btn_simulate = st.button("🚀 Jalankan Simulasi", type="primary")

# --- MAIN AREA ---
st.title("⛏️ Mining Value Chain Optimizer")
st.markdown("Sistem pendukung keputusan berbasis *Hybrid Simulation* & *GenAI*.")

# --- LOGIKA SIMULASI ---
if btn_simulate:
    with st.spinner("Sedang menjalankan simulasi hybrid (SimPy + ML)..."):
        # Siapkan Payload JSON
        payload = {
            "fixed_conditions": {
                "weatherCondition": weather,
                "roadCondition": road,
                "shift": shift,
                "target_road_id": "cmhsbjn8x02s2maft90hi31ty", # Hardcoded untuk demo
                "target_excavator_id": "cmhsbjpma05ddmaft5kv95dom"
            },
            "decision_variables": {
                "alokasi_truk": truck_opts,
                "jumlah_excavator": exc_opts
            },
            "financial_params": {
                "HargaJualBatuBara": price_coal,
                "HargaSolar": price_fuel,
                "BiayaPenaltiKeterlambatanKapal": cost_penalty
            }
        }
        
        try:
            # Panggil API
            response = requests.post(f"{API_URL}/get_top_3_strategies", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                # Simpan ke session state
                st.session_state.strategies_context = data['top_3_strategies']
                # Reset chat history saat simulasi baru
                st.session_state.chat_history = []
                st.success("Simulasi Selesai!")
            else:
                st.error(f"Gagal menghubungi API: {response.text}")
                
        except Exception as e:
            st.error(f"Error koneksi: {e}. Pastikan api.py sedang berjalan!")

# --- TAMPILAN HASIL (3 KOLOM) ---
if st.session_state.strategies_context:
    st.divider()
    st.subheader("📊 Rekomendasi Strategi Terbaik")
    
    cols = st.columns(3)
    
    for i, strat in enumerate(st.session_state.strategies_context):
        # Ambil data (handle nama key yang mungkin berbeda dari strategi ke strategi)
        # Di JSON kita, strukturnya: [{'STRATEGI_1': {...}}, {'STRATEGI_2': {...}}]
        key_name = list(strat.keys())[0]
        details = strat[key_name]
        
        # Tentukan warna kartu
        border = True if i == 0 else False # Highlight strategi 1
        
        with cols[i]:
            with st.container(border=True):
                # Judul
                if i == 0: st.markdown("### ⭐ Rekomendasi Utama")
                elif i == 1: st.markdown("### 🥈 Pilihan Efisien")
                else: st.markdown("### 🥉 Alternatif")
                
                # Metrik Utama
                st.metric("Estimasi Profit", details['KPI_PREDIKSI']['PROFIT'])
                
                # Detail
                st.markdown(f"**Fleet:** {details['INSTRUKSI_FLAT']['JUMLAH_DUMP_TRUCK']} & {details['INSTRUKSI_FLAT']['JUMLAH_EXCAVATOR']}")
                st.markdown(f"**Produksi:** {details['KPI_PREDIKSI']['PRODUKSI']}")
                st.markdown(f"**Fuel Ratio:** {details['KPI_PREDIKSI']['FUEL_RATIO']}")
                st.markdown(f"**Antrian:** {details['KPI_PREDIKSI']['IDLE_ANTRIAN']}")
                
                with st.expander("Lihat SOP & Instruksi"):
                    st.caption(details['SOP_KESELAMATAN'])
                    st.caption(f"Jalur: {details['INSTRUKSI_FLAT']['JALUR_ANGKUT']}")

    # --- BAGIAN CHATBOT (OLLAMA) ---
    st.divider()
    st.subheader("💬 Asisten Operasional (AI)")

    # Tampilkan riwayat chat
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input chat baru
    if prompt := st.chat_input("Tanya detail tentang strategi di atas..."):
        # 1. Tampilkan pertanyaan user
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Panggil API Chatbot
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("▌ Sedang berpikir...")
            
            try:
                chat_payload = {
                    "pertanyaan_user": prompt,
                    "top_3_strategies_context": st.session_state.strategies_context
                }
                res = requests.post(f"{API_URL}/ask_chatbot", json=chat_payload)
                
                if res.status_code == 200:
                    ai_response = res.json()['jawaban_ai']
                    message_placeholder.markdown(ai_response)
                    st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
                else:
                    message_placeholder.error("Gagal mendapatkan respon dari AI.")
            except Exception as e:
                message_placeholder.error(f"Error koneksi: {e}")

else:
    st.info("👈 Silakan atur parameter di sidebar dan klik 'Jalankan Simulasi' untuk memulai.")