from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import pandas as pd
from typing import List, Dict, Any, Optional
import ollama

# --- 1. Impor "Otak" Anda dari simulator.py ---
try:
    from simulator import (
        CONFIG,  # Konfigurasi default
        get_strategic_recommendations, # Agen 1 (Hybrid)
        format_konteks_for_llm,      # Helper untuk menerjemahkan data ke AI
        LLM_PROVIDER,                # Untuk mengecek apakah Ollama terhubung
        OLLAMA_MODEL                 # Nama model Ollama
    )
    print("✅ Berhasil mengimpor 'otak' dari simulator.py")
except ImportError as e:
    print(f"❌ ERROR: Gagal mengimpor dari 'simulator.py'. Pastikan file ada. Detail: {e}")
    exit()
except Exception as e:
    print(f"❌ ERROR saat inisialisasi simulator.py: {e}")
    exit()

# --- 2. Inisialisasi Aplikasi API ---
app = FastAPI(
    title="Mining Ops API (Arsitektur Hybrid + Ollama)",
    description="API untuk Agen Strategis (Top 3) dan Agen Chatbot (Follow-up) lokal."
)

# --- 3. Tentukan Model Input (Kontrak Data) ---

# Model BARU untuk parameter finansial kustom
class FinancialParams(BaseModel):
    HargaJualBatuBara: float = Field(800000, description="Harga Jual per Ton")
    HargaSolar: float = Field(15000, description="Harga Solar per Liter")
    BiayaPenaltiKeterlambatanKapal: float = Field(100000000, description="Biaya Penalti per Jam Antri")

class FixedConditions(BaseModel):
    weatherCondition: str = "Cerah"
    roadCondition: str = "GOOD"
    shift: str = "SHIFT_1"
    target_road_id: str
    target_excavator_id: str

class DecisionVariables(BaseModel):
    alokasi_truk: List[int] = [5, 10]
    jumlah_excavator: List[int] = [1, 2]

class RecommendationRequest(BaseModel):
    """
    Request body lengkap untuk endpoint strategi.
    Termasuk 'financial_params' yang opsional.
    """
    fixed_conditions: FixedConditions
    decision_variables: DecisionVariables
    # --- PERUBAHAN DI SINI ---
    financial_params: Optional[FinancialParams] = None # <-- KUNCI UNTUK WHAT-IF

class ChatRequest(BaseModel):
    pertanyaan_user: str
    top_3_strategies_context: List[Dict[str, Any]]

# --- 4. Buat Endpoint API ---

@app.post("/get_top_3_strategies")
async def dapatkan_rekomendasi_strategis(request: RecommendationRequest):
    """
    ENDPOINT AGEN 1 (STRATEGIS - HYBRID):
    Menerima kondisi lapangan, variabel keputusan,
    dan parameter finansial kustom (opsional).
    """
    try:
        # --- LOGIKA BARU UNTUK MEMILIH PARAMETER ---
        active_financial_params = {}
        if request.financial_params:
            # 1. Jika FE mengirim parameter kustom, gunakan itu
            active_financial_params = request.financial_params.dict()
            print("Info: Menjalankan simulasi dengan parameter finansial kustom dari FE.")
        else:
            # 2. Jika tidak, gunakan default dari server
            active_financial_params = CONFIG['financial_params']
            print("Info: Menjalankan simulasi dengan parameter finansial default.")
        
        # Panggil "otak" AI dengan parameter finansial yang aktif
        top_3_list = get_strategic_recommendations(
            request.fixed_conditions.dict(),
            request.decision_variables.dict(),
            active_financial_params # Berikan parameter finansial ke mesin
        )
        
        if top_3_list:
            return {
                "top_3_strategies": top_3_list
            }
        else:
            raise HTTPException(status_code=500, detail="Gagal menjalankan optimisasi strategi.")
    except Exception as e:
        print(f"Error di /get_top_3_strategies: {e}")
        raise HTTPException(status_code=500, detail=f"Error internal: {str(e)}")


@app.post("/ask_chatbot")
async def tanya_jawab_chatbot(request: ChatRequest):
    """
    ENDPOINT AGEN 2 (CHATBOT) - Ditenagai OLLAMA
    (Endpoint ini tidak perlu diubah)
    """
    
    if LLM_PROVIDER != "ollama":
        raise HTTPException(status_code=503, detail="Layanan Chatbot (Ollama) tidak terhubung di server.")

    try:
        data_konteks_string = format_konteks_for_llm(request.top_3_strategies_context)
        
        system_prompt = f"""
        !!! PERINTAH UTAMA: RESPONS ANDA HARUS SELALU DALAM BAHASA INDONESIA. !!!
        PERAN ANDA: Asisten Analis Operasi Tambang...
        DATA 3 STRATEGI TERBAIK (JSON):
        {data_konteks_string}
        ATURAN:
        1. Jawab HANYA berdasarkan 3 strategi dalam DATA KONTEKS di atas.
        2. Fokus pada 'ESTIMASI_PROFIT_SHIFT'.
        3. Gunakan Bahasa Indonesia.
        """
        
        messages_for_ollama = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': request.pertanyaan_user}
        ]
        
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=messages_for_ollama
        )
        
        jawaban_ai = response['message']['content']
        
        return {"jawaban_ai": jawaban_ai}

    except Exception as e:
        if "Connection refused" in str(e):
             raise HTTPException(status_code=503, detail="Layanan Chatbot (Ollama) tidak berjalan di server.")
        print(f"Error di /ask_chatbot: {e}")
        raise HTTPException(status_code=500, detail=f"Error saat menghubungi Ollama: {str(e)}")

# --- 5. Jalankan Server API ---
if __name__ == "__main__":
    print("Menjalankan Uvicorn server di http://127.0.0.1:8000")
    print("Buka http://127.0.0.1:8000/docs untuk melihat dokumentasi API.")
    uvicorn.run(app, host="127.0.0.1", port=8000)