import os
import json
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI

# .env içinden anahtarı yükle
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY bulunamadı. .env dosyasını kontrol et.")

client = OpenAI(api_key=api_key)

app = FastAPI()

# CORS – Flutter rahatça erişsin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(request: Request):
    """
    Flutter NormalAnalizScreen'den gelen isteği karşılar.
    Burada HİÇ Pydantic / Body validation yok -> 422 atacak kimse kalmıyor.
    """

    # 🔹 Gövdeyi kendimiz parse ediyoruz, hata olursa boş dict'e düşer
    try:
      body = await request.json()
    except Exception:
      body = {}

    if not isinstance(body, dict):
      body = {}

    vehicle: Dict[str, Any] = body.get("vehicle") or {}
    profile: Dict[str, Any] = body.get("profile") or {}
    ad_description: Optional[str] = body.get("ad_description") or ""
    screenshot_base64: Optional[str] = body.get("screenshot_base64")

    make = (vehicle.get("make") or "").strip() or "Bilinmiyor"
    model = (vehicle.get("model") or "").strip() or "Bilinmiyor"
    year = vehicle.get("year")
    mileage = vehicle.get("mileage_km")
    fuel = vehicle.get("fuel")

    yearly_km = profile.get("yearly_km")
    usage = profile.get("usage")
    fuel_pref = profile.get("fuel_preference")

    ad_text = ad_description or "İlan açıklaması verilmemiş."

    base_text = f"""
Kullanıcı Türkiye'de ikinci el araç bakıyor. Araç ve profil bilgileri:

Araç:
- Marka / Model: {make} {model}
- Model yılı: {year or 'bilinmiyor'}
- Kilometre: {mileage or 'bilinmiyor'} km
- Yakıt türü: {fuel or 'bilinmiyor'}

Kullanıcı profili:
- Kullanım tipi: {usage or 'belirtilmemiş'} (city/mixed/highway)
- Yıllık km: {yearly_km or 'bilinmiyor'}
- Yakıt tercihi: {fuel_pref or 'belirtilmemiş'}

İlan açıklaması:
\"\"\"{ad_text}\"\"\".

Bu bilgilerle aracı değerlendir; Türkiye şartlarına göre konuş.
"""

    contents: List[Dict[str, Any]] = [
        {
            "type": "input_text",
            "text": base_text,
        }
    ]

    if screenshot_base64:
        contents.append(
            {
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{screenshot_base64}",
            }
        )

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "car_analysis",
            "schema": {
                "type": "object",
                "properties": {
                    "scores": {
                        "type": "object",
                        "properties": {
                            "overall_100": {"type": "integer"},
                            "reliability_100": {"type": "integer"},
                            "running_cost_100": {"type": "integer"},
                            "parts_availability_100": {"type": "integer"},
                            "suitability_100": {"type": "integer"},
                        },
                        "required": ["overall_100"],
                    },
                    "summary": {
                        "type": "object",
                        "properties": {
                            "short_text": {"type": "string"},
                            "pros": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "cons": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["short_text", "pros", "cons"],
                    },
                },
                "required": ["scores", "summary"],
            },
        },
    }

    try:
        resp = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "Sen Türkiye'deki ikinci el araç piyasasını iyi bilen, "
                                "oto ekspertiz + finans uzmanı bir asistansın. "
                                "Kullanıcının bütçesine ve kullanımına göre yorum yap. "
                                "Sadece JSON formatında cevap döndür."
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": contents,
                },
            ],
            response_format=response_format,
        )

        raw = resp.output[0].content[0].text

        try:
            data = json.loads(raw)
        except Exception as parse_err:
            raise HTTPException(
                status_code=500,
                detail=f"Model JSON'u parse edilemedi: {parse_err}. Raw: {raw[:200]}",
            )

        return data

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analiz sırasında bir hata oluştu: {e}",
        )
