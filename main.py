import os
from urllib.parse import urlparse
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CarRequest(BaseModel):
    url: str
    budget: str

# ---------------- URL'DEN ARAÇ ADI ÇIKARTMA ----------------

def extract_car_name_from_url(url: str) -> str:
    try:
        parsed = urlparse(url)
        path = parsed.path
        segments = [s for s in path.split("/") if s]

        if not segments:
            return ""

        last = segments[-1]
        parts = last.split("-")

        # son bölüm id ise temizle
        if parts and parts[-1].isdigit():
            parts = parts[:-1]

        text = " ".join(parts)
        text = text.replace(".", "").replace("-", " ")

        return text.strip()
    except:
        return ""

# -----------------------------------------------------------

@app.post("/analyze")
def analyze_car(data: CarRequest):

    guessed = extract_car_name_from_url(data.url)
    if not guessed:
        guessed = "Bilinmeyen araç"

    prompt = f"""
Sen Türkiye'de ikinci el araçlar konusunda uzman bir oto eksperisin.

Kullanıcıdan gelen veriler:
- İlan linki: {data.url}
- URL tahmini araç adı: {guessed}
- Bütçe: {data.budget} TL

ARAÇ SEGMENT KURALLARI:
- Passat -> D segment
- BMW 5 Serisi -> E segment
- BMW 3 Serisi -> D segment
- Corolla / Egea / Megane -> C segment
- Clio / Corsa -> B segment
- Emin değilsen "muhtemelen X segmenti" de, ama yanlış segment söyleme.

ANALİZ YAPISI:

1) ÖZET KARAR:
   İlk satırda net şekilde “ALINIR” veya “ALINMAZ” yaz,
   yanına kısa gerekçe ekle.

2) SEGMENT VE ARAÇ TİPİ:
   - Sedan / HB / SUV tahmini
   - Segment (kurallara göre)
   - Bu segmentin kullanım amacı

3) TEKNİK ANALİZ:
   - Motor tarafı (riskler, tipik sorunlar)
   - Yürüyen (aks, amortisör, fren, direksiyon)
   - Şasi / airbag / pert ihtimalleri
   - Kilometre şüphesi olasılıkları
   - Kronik model sorunları

4) FİYAT + BÜTÇE:
   - Tahmini piyasa aralığı
   - Bu bütçeyle alınır mı?
   - Alternatif araç önerileri

5) SON SATIR:
   “Karar: ALINIR”
   veya
   “Karar: ALINMAZ”

Klişe cümleleri azalt, tekrar yapma, araca özel yorum üret.
"""

    try:
        response = client.responses.create(
            model="gpt-4o-mini",
            input=prompt
        )
        text = response.output_text
    except Exception as e:
        text = f"API hatası: {e}"

    return {"analysis": text}
