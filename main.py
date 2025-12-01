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
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # İleride domainlere göre kısıtlayabilirsin.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# Modeller
# -------------------------------------------------

class AnalyzeRequest(BaseModel):
    text: str


class AnalyzeResponse(BaseModel):
    analysis: str


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-4o-mini"


def _call_openai(system_prompt: str, user_content: str) -> str:
    if not OPENAI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY tanımlı değil. Lütfen Render ortam değişkenlerine ekle.",
        )

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.7,
    }

    try:
        resp = requests.post(OPENAI_API_URL, json=payload, headers=headers, timeout=60)
    except requests.RequestException as e:
        raise HTTPException(
            status_code=502,
            detail=f"OpenAI isteği sırasında bağlantı hatası oluştu: {e}",
        )

    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"OpenAI hatası ({resp.status_code}): {resp.text}",
        )

    data = resp.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="OpenAI cevabı beklenmedik formatta geldi.",
        )

    return content.strip()


def generate_normal_analysis(text: str) -> str:
    system_prompt = (
        "Sen uzman bir oto ekspertörüsün. Gönderilen ilan metnini dikkatlice okuyup, "
        "ikinci el araç alıcısına anlaşılır bir dille genel bir analiz ver.\n\n"
        "Analizde kısaca şu başlıklara değin:\n"
        "- Araç tipi ve kullanım amacı\n"
        "- Motor, şanzıman, kilometre ve yaş durumu\n"
        "- Yakıt tipi ve tüketim beklentisi\n"
        "- Tramer / boya / değişen durumu\n"
        "- Kısaca fiyat/fayda ve alınır mı alınmaz mı yorumu\n\n"
        "Net ve samimi Türkçe kullan. Çok uzun yazma, sade ve anlaşılır olsun."
    )

    user_content = f"Aşağıdaki ilan metnine göre NORMAL analiz yap:\n\n{text}"
    return _call_openai(system_prompt, user_content)


def generate_premium_analysis(text: str) -> str:
    system_prompt = (
        "Sen deneyimli bir oto ekspertör ve ikinci el araç danışmanısın. "
        "Gönderilen ilan metnine göre detaylı, PREMIUM bir analiz hazırla.\n\n"
        "Aşağıdaki başlıkların HER BİRİNE ayrı ayrı değin:\n"
        "1) Araç Profili ve Kullanım Amacı\n"
        "2) Motor & Şanzıman & Kilometre Değerlendirmesi\n"
        "3) Yakıt Tipi, Yakıt Tüketimi ve Güncel Yakıt Fiyatlarına Göre Maliyet\n"
        "4) Tramer, Boya, Değişen ve Kaza Riski\n"
        "5) İlan Sahibinin Yazdıkları: Güvenilirlik ve Dikkat Çeken Noktalar\n"
        "6) Yakın Vadede Çıkabilecek Muhtemel Masraflar (parça/parça yaz)\n"
        "7) Ortalama Piyasa Değeri Tahmini (net fiyat söyleme, aralık ver)\n"
        "8) Fiyat/Fayda Yorumu ve Alınır mı?/Alınmaz mı? şeklinde net bir sonuç\n\n"
        "Premium kullanıcıya hitap eder gibi daha detaylı, ama yine de anlaşılır Türkçe kullan. "
        "Gerektiği yerde uyarıcı ol ama boş yere korkutma. Madde madde ve başlıklarla yaz."
    )

    user_content = f"Aşağıdaki ilan metnine göre PREMIUM analiz yap:\n\n{text}"
    return _call_openai(system_prompt, user_content)


# -------------------------------------------------
# Endpointler
# -------------------------------------------------

@app.get("/")
async def root():
    return {"message": "Oto Analiz backend çalışıyor. /analyze ve /premium_analyze endpointlerini kullan."}


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    """
    NORMAL analiz endpoint'i.
    Gövde: {"text": "... ilan açıklaması ..."}
    """
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Metin boş olamaz.")
    analysis = generate_normal_analysis(req.text)
    return AnalyzeResponse(analysis=analysis)


@app.post("/premium_analyze", response_model=AnalyzeResponse)
async def premium_analyze(req: AnalyzeRequest):
    """
    PREMIUM analiz endpoint'i.
    Gövde yine: {"text": "... ilan açıklaması ..."}
    """
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Metin boş olamaz.")
    analysis = generate_premium_analysis(req.text)
    return AnalyzeResponse(analysis=analysis)
