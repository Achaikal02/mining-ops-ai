import uvicorn
import ollama
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# --- 1. IMPOR DARI SIMULATOR ---
try:
    from simulator import (
        CONFIG, 
        get_strategic_recommendations, 
        format_konteks_for_llm, # Fungsi format agar JSON cantik
        LLM_PROVIDER, 
        OLLAMA_MODEL 
    )
    print("✅ Berhasil mengimpor 'otak' dari simulator.py")
except ImportError as e:
    print(f"❌ ERROR CRITICAL: Gagal mengimpor dari 'simulator.py'. Detail: {e}")
    exit()
except Exception as e:
    print(f"❌ ERROR saat inisialisasi simulator: {e}")
    exit()

app = FastAPI(title="Mining Ops AI API", version="2.0.0")

# --- MODEL INPUT ---
class FinancialParams(BaseModel):
    HargaJualBatuBara: float = Field(800000)
    HargaSolar: float = Field(15000)
    BiayaPenaltiKeterlambatanKapal: float = Field(100000000)
    BiayaRataRataInsiden: float = Field(50000000)

class FixedConditions(BaseModel):
    weatherCondition: str
    roadCondition: str
    shift: str
    target_road_id: str
    target_excavator_id: str

class DecisionVariables(BaseModel):
    alokasi_truk: List[int]
    jumlah_excavator: List[int]

class RecommendationRequest(BaseModel):
    fixed_conditions: FixedConditions
    decision_variables: DecisionVariables
    financial_params: Optional[FinancialParams] = None 

class ChatRequest(BaseModel):
    pertanyaan_user: str
    top_3_strategies_context: List[Dict[str, Any]]

# --- ENDPOINTS ---

@app.post("/get_top_3_strategies")
async def dapatkan_rekomendasi_strategis(request: RecommendationRequest):
    try:
        print(f"📡 Menerima request strategi baru...")
        
        active_financial_params = {}
        if request.financial_params:
            active_financial_params = request.financial_params.dict()
            print("   ℹ️ Menggunakan Parameter Finansial Kustom")
        else:
            active_financial_params = CONFIG['financial_params']
            print("   ℹ️ Menggunakan Parameter Finansial Default")
        
        # 1. JALANKAN SIMULASI
        top_3_list = get_strategic_recommendations(
            request.fixed_conditions.dict(),
            request.decision_variables.dict(),
            active_financial_params 
        )
        
        if top_3_list:
            # 2. FORMAT DATA AGAR COCOK DENGAN DASHBOARD (PENTING!)
            # Ini mengubah data mentah menjadi struktur dengan 'KPI_PREDIKSI', 'INSTRUKSI_FLAT' dll
            formatted_json_str = format_konteks_for_llm(top_3_list)
            formatted_data = json.loads(formatted_json_str)
            
            return {"top_3_strategies": formatted_data}
        else:
            raise HTTPException(status_code=500, detail="Simulasi selesai tapi tidak menghasilkan rekomendasi valid.")
            
    except Exception as e:
        print(f"❌ Error di /get_top_3_strategies: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.post("/ask_chatbot")
async def tanya_jawab_chatbot(request: ChatRequest):
    if LLM_PROVIDER != "ollama":
        raise HTTPException(status_code=503, detail="Layanan Chatbot (Ollama) tidak terhubung.")

    try:
        # Gunakan data yang dikirim dari dashboard (sudah terformat)
        # Kita dump kembali ke string untuk prompt
        data_konteks_string = json.dumps(request.top_3_strategies_context, indent=2)
        
        system_prompt = f"""
        !!! PERINTAH UTAMA: RESPONS ANDA HARUS SELALU DALAM BAHASA INDONESIA. !!!
        PERAN ANDA: Kepala Teknik Tambang (KTT).
        DATA OPERASIONAL:
        {data_konteks_string}
        ATURAN:
        1. Jawab HANYA berdasarkan data di atas.
        2. Fokus pada Profit, Fuel Ratio, dan Efisiensi.
        3. Gunakan Bahasa Indonesia profesional.
        """
        
        messages_for_ollama = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': request.pertanyaan_user}
        ]
        
        response = ollama.chat(model=OLLAMA_MODEL, messages=messages_for_ollama)
        return {"jawaban_ai": response['message']['content']}

    except Exception as e:
        print(f"❌ Error Chatbot: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error Chatbot: {str(e)}")

if __name__ == "__main__":
    print("🚀 Memulai Server API...")
    uvicorn.run(app, host="127.0.0.1", port=8000)