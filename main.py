import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# -------------------------------------------------
# FastAPI uygulaması + CORS ayarları
# -------------------------------------------------

app = FastAPI(
    title="Oto Analiz Backend",
    description="İlan metnine göre ikinci el araç analizi yapan servis.",
    version="1.0.0",
)

# Burada şimdilik her yerden isteğe izin veriyoruz.
# İstersen ileride sadece kendi domainlerini yazarsın.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Örn: ["https://otoanaliz.app", "http://localhost:62635"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# Modeller
# -------------------------------------------------

class AnalyzeRequest(BaseModel):
    text: str   # Flutter bu key ile gönderiyor: {"text": "ilan açıklaması..."}


class AnalyzeResponse(BaseModel):
    analysis: str


# -------------------------------------------------
# Yardımcı fonksiyon: OpenAI'den analiz alma
# -------------------------------------------------

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-4o-mini"  # İstersen gpt-4o, gpt-3.5-turbo vs yapabilirsin.


def generate_car_analysis(text: str) -> str:
    """
    İlan açıklamasına göre detaylı ikinci el araç analizi yapan fonksiyon.
    OpenAI Chat Completions API'sini direkt HTTP isteği ile kullanıyoruz.
    """

    if not OPENAI_API_KEY:
        # Backend ayarları eksikse anlamlı bir hata gönder.
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY tanımlı değil. Lütfen Render ortam değişkenlerine ekle.",
        )

    system_prompt = (
        "Sen uzman bir oto ekspertörüsün. Gönderilen ilan metnini dikkatlice okuyup, "
        "ikinci el araç alıcısına anlaşılır bir dille detaylı analiz ver.\n\n"
        "Analizde şu başlıklara değin:\n"
        "- Araç tipi ve kullanım amacı (şehir içi, aile, uzun yol, performans vs.)\n"
        "- Motor, şanzıman, kilometre ve yaş durumuna göre olası riskler\n"
        "- Yakıt tipi ve tüketim beklentisi\n"
        "- Tramer, boya, değişen durumuna göre kasa durumu\n"
        "- İlan sahibinin yazdığı artı/eksi noktaları değerlendir\n"
        "- Ortalama piyasa şartlarında bu aracın mantıklı olup olmadığı\n"
        "- Fiyat/fayda ve uzun vadeli masraf beklentisi\n\n"
        "Net, samimi ve anlaşılır Türkçe kullan. Gerektiğinde uyar ama gereksiz korkutma."
    )

    user_prompt = (
        "Aşağıdaki ilan metnine göre analiz yap:\n\n"
        f"--- İLAN METNİ ---\n{text}\n------------------"
    )

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.7,
    }

    try:
        response = requests.post(OPENAI_API_URL, json=payload, headers=headers, timeout=60)
    except requests.RequestException as e:
        raise HTTPException(
            status_code=502,
            detail=f"OpenAI isteği sırasında bağlantı hatası oluştu: {e}",
        )

    if response.status_code != 200:
        # OpenAI tarafındaki hatayı da iletelim ki teşhis kolay olsun
        raise HTTPException(
            status_code=502,
            detail=f"OpenAI hatası ({response.status_code}): {response.text}",
        )

    data = response.json()
    try:
        analysis_text = data["choices"][0]["message"]["content"]
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="OpenAI cevabı beklenmedik formatta geldi.",
        )

    return analysis_text.strip()


# -------------------------------------------------
# Endpointler
# -------------------------------------------------

@app.get("/")
async def root():
    return {"message": "Oto Analiz backend çalışıyor. /analyze endpointini kullan."}


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    """
    Flutter uygulamasının POST attığı ana endpoint.
    Gövde: {"text": "... ilan açıklaması ..."}
    Cevap: {"analysis": "... detaylı analiz ..."}
    """
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Metin boş olamaz.")

    analysis = generate_car_analysis(req.text)
    return AnalyzeResponse(analysis=analysis)
