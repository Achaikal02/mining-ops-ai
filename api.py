import uvicorn
import ollama
import json
import pandas as pd
import os
import uuid
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# --- 1. IMPOR "OTAK" AI DARI SIMULATOR.PY ---
# Ini akan memuat model ML dan database ke memori saat server start
try:
    from simulator import (
        CONFIG,                        # Konfigurasi Default
        get_strategic_recommendations, # Fungsi Agen Strategis (Hybrid)
        format_konteks_for_llm,        # Helper formatter data
        LLM_PROVIDER,                  # Cek status Ollama
        OLLAMA_MODEL                   # Nama model (qwen2.5:7b / llama3:8b)
    )
    print("✅ Berhasil mengimpor 'otak' dari simulator.py")
except ImportError as e:
    print(f"❌ ERROR CRITICAL: Gagal mengimpor dari 'simulator.py'.")
    print(f"Pastikan file 'simulator.py' ada di folder yang sama.")
    print(f"Detail Error: {e}")
    exit()
except Exception as e:
    print(f"❌ ERROR saat inisialisasi simulator: {e}")
    exit()

# --- 2. INISIALISASI APLIKASI API ---
app = FastAPI(
    title="Mining Ops AI API",
    description="Backend API untuk simulasi tambang hybrid, logistik kapal, dan chatbot KTT (Ollama).",
    version="3.0.0"
)

# --- 3. DEFINISI MODEL INPUT (DATA CONTRACT) ---

# Model untuk Parameter Ekonomi (What-If Analysis)
class FinancialParams(BaseModel):
    """Parameter Finansial Kustom (Opsional)"""
    HargaJualBatuBara: float = Field(800000, description="Harga Jual per Ton (IDR)")
    HargaSolar: float = Field(15000, description="Harga Solar per Liter (IDR)")
    BiayaPenaltiKeterlambatanKapal: float = Field(100000000, description="Biaya Penalti per Jam Antri (IDR)")
    BiayaRataRataInsiden: float = Field(50000000, description="Biaya rata-rata per insiden (IDR)")
    BiayaDemurragePerJam: float = Field(50000000, description="Denda kapal per jam (IDR)")

# Model untuk Kondisi Lapangan
class FixedConditions(BaseModel):
    """Input Kondisi Lapangan (Tetap)"""
    weatherCondition: str = Field("Cerah", example="Hujan Ringan")
    roadCondition: str = Field("GOOD", example="FAIR")
    shift: str = Field("SHIFT_1", example="SHIFT_1")
    target_road_id: str = Field("cmhsbjn8x02s2maft90hi31ty", description="ID Rute dari database")
    target_excavator_id: str = Field("cmhsbjpma05ddmaft5kv95dom", description="ID Excavator dari database")
    target_schedule_id: Optional[str] = Field(None, description="ID Jadwal Kapal (Opsional)")
    simulation_start_date: Optional[str] = Field(None, description="Tanggal mulai simulasi (ISO 8601)")

# Model untuk Variabel Keputusan
class DecisionVariables(BaseModel):
    """Input Variabel Keputusan (Yang ingin diuji)"""
    alokasi_truk: List[int] = Field([5, 10], description="Daftar jumlah truk untuk disimulasikan")
    jumlah_excavator: List[int] = Field([1, 2], description="Daftar jumlah excavator untuk disimulasikan")

# Model Request Utama (Simulasi)
class RecommendationRequest(BaseModel):
    """Payload utama untuk Endpoint Strategi"""
    fixed_conditions: FixedConditions
    decision_variables: DecisionVariables
    # Opsional: Jika user tidak mengirim ini, server pakai default
    financial_params: Optional[FinancialParams] = None 

# Model Request Chatbot
class ChatRequest(BaseModel):
    """Payload untuk Endpoint Chatbot"""
    pertanyaan_user: str
    top_3_strategies_context: List[Dict[str, Any]] # Konteks strategi dari request sebelumnya

# Model untuk Menambah Data Baru (Kapal/Jadwal)
class NewVessel(BaseModel):
    name: str
    type: str = "Bulk Carrier"
    capacity: float

class NewSchedule(BaseModel):
    vesselId: str
    loadingPort: str
    destination: str
    etaLoading: str # Format ISO
    etsLoading: str # Deadline
    plannedQuantity: float
    buyer: str

# --- 4. ENDPOINT API UTAMA ---

@app.get("/")
def read_root():
    return {"status": "online", "service": "Mining Ops AI Assistant v3.0"}

@app.post("/get_top_3_strategies")
async def dapatkan_rekomendasi_strategis(request: RecommendationRequest):
    """
    ENDPOINT 1: AGEN STRATEGIS (Simulasi Hybrid)
    Menerima input user, menjalankan simulasi SimPy+ML, dan mengembalikan 3 strategi terbaik
    yang sudah diformat rapi untuk ditampilkan di Dashboard.
    """
    try:
        print(f"📡 Menerima request strategi baru...")
        
        # 1. Tentukan Parameter Finansial (Default vs Kustom)
        active_financial_params = {}
        if request.financial_params:
            active_financial_params = request.financial_params.dict()
            print("   ℹ️ Menggunakan Parameter Finansial Kustom (User Input)")
        else:
            active_financial_params = CONFIG['financial_params']
            print("   ℹ️ Menggunakan Parameter Finansial Default Server")
        
        # 2. Panggil Mesin Simulasi di simulator.py
        top_3_list = get_strategic_recommendations(
            request.fixed_conditions.dict(),
            request.decision_variables.dict(),
            active_financial_params # Berikan parameter finansial ke mesin
        )
        
        if top_3_list:
            # 3. Format Data (PENTING UNTUK DASHBOARD)
            # Mengubah data mentah menjadi struktur cantik (KPI_PREDIKSI, INSTRUKSI_FLAT)
            formatted_json_str = format_konteks_for_llm(top_3_list)
            formatted_data = json.loads(formatted_json_str)
            
            return {"top_3_strategies": formatted_data}
        else:
            # Jika simulasi gagal menemukan solusi valid
            raise HTTPException(status_code=500, detail="Simulasi selesai tapi tidak menghasilkan rekomendasi valid.")
            
    except Exception as e:
        print(f"❌ Error di /get_top_3_strategies: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


@app.post("/ask_chatbot")
async def tanya_jawab_chatbot(request: ChatRequest):
    """
    ENDPOINT 2: AGEN CHATBOT (Ollama Local)
    Menjawab pertanyaan user berdasarkan konteks 3 strategi terbaik.
    """
    
    # Cek koneksi Ollama
    if LLM_PROVIDER != "ollama":
        raise HTTPException(status_code=503, detail="Layanan Chatbot (Ollama) tidak terhubung di server.")

    try:
        print(f"💬 Menerima pertanyaan chatbot: {request.pertanyaan_user}")

        # 1. Format data ulang untuk prompt (memastikan struktur string json rapi)
        # Karena request.top_3_strategies_context sudah berupa list dict dari frontend,
        # kita dump kembali ke string JSON agar bisa dimasukkan ke prompt teks.
        data_konteks_string = json.dumps(request.top_3_strategies_context, indent=2)
        
        # 2. System Prompt (Persona KTT)
        system_prompt = f"""
        !!! PERINTAH UTAMA: RESPONS ANDA HARUS SELALU DALAM BAHASA INDONESIA. !!!
        
        PERAN ANDA:
        Anda adalah KEPALA TEKNIK TAMBANG (KTT) senior yang berpengalaman.
        Tugas Anda adalah menjawab pertanyaan user mengenai 3 STRATEGI TERBAIK yang datanya diberikan di bawah ini.

        DATA 3 STRATEGI TERBAIK (JSON):
        {data_konteks_string}
        
        ATURAN MENJAWAB:
        1. Jawab HANYA berdasarkan data konteks di atas. Jangan halusinasi.
        2. Fokus jawaban Anda pada 3 strategi ini saja.
        3. Gunakan istilah 'ESTIMASI_PROFIT', 'FUEL_RATIO' (L/Ton), dan 'IDLE_ANTRIAN'.
        4. Jika ditanya rekomendasi, jelaskan trade-off antara Profit vs Efisiensi/Antrian.
        5. Gunakan Bahasa Indonesia yang profesional, tegas, dan teknis.
        """
        
        # 3. Kirim ke Ollama (Stateless Request)
        messages_for_ollama = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': request.pertanyaan_user}
        ]
        
        response = ollama.chat(
            model=OLLAMA_MODEL, # Menggunakan model yang didefinisikan di simulator.py
            messages=messages_for_ollama
        )
        
        jawaban_ai = response['message']['content']
        
        return {"jawaban_ai": jawaban_ai}

    except Exception as e:
        # Handle jika Ollama mati mendadak
        if "Connection refused" in str(e):
             raise HTTPException(status_code=503, detail="Gagal menghubungi Ollama. Pastikan server Ollama berjalan.")
        print(f"❌ Error di /ask_chatbot: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error Chatbot: {str(e)}")

# --- 5. ENDPOINT MANAJEMEN DATA (BONUS) ---

@app.post("/add_vessel")
async def add_vessel(vessel: NewVessel):
    """Menambah data master kapal ke CSV"""
    try:
        file_path = os.path.join('data', 'vessels.csv')
        new_id = f"ves-{uuid.uuid4().hex[:8]}"
        
        new_data = pd.DataFrame([{
            "id": new_id,
            "name": vessel.name,
            "type": vessel.type,
            "capacity": vessel.capacity,
            "createdAt": datetime.now().isoformat()
        }])
        
        if os.path.exists(file_path):
            new_data.to_csv(file_path, mode='a', header=False, index=False)
        else:
            new_data.to_csv(file_path, mode='w', header=True, index=False)
            
        return {"status": "success", "message": f"Kapal {vessel.name} berhasil ditambahkan.", "id": new_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add_schedule")
async def add_schedule(schedule: NewSchedule):
    """Menambah jadwal pengapalan baru ke CSV"""
    try:
        file_path = os.path.join('data', 'sailing_schedules.csv')
        new_id = f"sch-{uuid.uuid4().hex[:8]}"
        
        new_data = pd.DataFrame([{
            "id": new_id,
            "vesselId": schedule.vesselId,
            "loadingPort": schedule.loadingPort,
            "destination": schedule.destination,
            "etaLoading": schedule.etaLoading,
            "etsLoading": schedule.etsLoading,
            "plannedQuantity": schedule.plannedQuantity,
            "actualQuantity": 0,
            "buyer": schedule.buyer,
            "status": "SCHEDULED",
            "createdAt": datetime.now().isoformat()
        }])
        
        if os.path.exists(file_path):
            new_data.to_csv(file_path, mode='a', header=False, index=False)
        else:
            new_data.to_csv(file_path, mode='w', header=True, index=False)
            
        return {"status": "success", "message": "Jadwal berhasil dibuat.", "id": new_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- 6. JALANKAN SERVER ---
if __name__ == "__main__":
    print("🚀 Memulai Server API...")
    print("📄 Dokumentasi tersedia di: http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="127.0.0.1", port=8000)