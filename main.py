import os
import json
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------
# Ortam değişkenleri ve OpenAI client
# ---------------------------------------------------------
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY bulunamadı. .env dosyasını kontrol et.")

client = OpenAI(api_key=api_key)

OPENAI_MODEL_DEFAULT = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_MODEL_NORMAL = os.getenv("OPENAI_MODEL_NORMAL", OPENAI_MODEL_DEFAULT)
OPENAI_MODEL_PREMIUM = os.getenv("OPENAI_MODEL_PREMIUM", OPENAI_MODEL_DEFAULT)
OPENAI_MODEL_COMPARE = os.getenv("OPENAI_MODEL_COMPARE", OPENAI_MODEL_DEFAULT)
OPENAI_MODEL_OTOBOT = os.getenv("OPENAI_MODEL_OTOBOT", OPENAI_MODEL_DEFAULT)

# ---------------------------------------------------------
# FastAPI app (Render bunu kullanıyor: uvicorn main:app)
# ---------------------------------------------------------
app = FastAPI(title="Oto Analiz Backend")

app.add_middleware(
    CORSMiddleware(
        allow_origins=["*"],      # İleride domain ile kısıtlayabiliriz
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
)

# ---------------------------------------------------------
# Pydantic modeller
# ---------------------------------------------------------

class Profile(BaseModel):
    # Default veriyoruz ki eksik gelse bile 422 patlamasın.
    yearly_km: int = Field(15000, ge=0, le=100000)
    # Serbest string; beklediğimiz: "city", "mixed", "highway"
    usage: str = "mixed"
    # Serbest string; beklediğimiz: "gasoline", "diesel", "lpg", "hybrid", "electric"
    fuel_preference: str = "gasoline"


class Vehicle(BaseModel):
    make: str = ""
    model: str = ""
    year: Optional[int] = Field(None, ge=1980, le=2035)
    mileage_km: Optional[int] = Field(None, ge=0)
    fuel: Optional[str] = None  # "gasoline" | "diesel" | "lpg" | ...


class AnalyzeRequest(BaseModel):
    profile: Profile = Field(default_factory=Profile)
    vehicle: Vehicle = Field(default_factory=Vehicle)

    # Eski sürümle uyum: tek screenshot string'i de kabul et
    screenshot_base64: Optional[str] = None
    screenshots_base64: Optional[List[str]] = None

    ad_description: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        # İleride Flutter'dan ekstra bir şey gönderirsen kırılmasın
        extra = "allow"


# Karşılaştırma için basit model – istersek genişletiriz
class CompareSide(BaseModel):
    title: Optional[str] = None      # Örn: "Corolla 1.6"
    vehicle: Vehicle = Field(default_factory=Vehicle)
    ad_description: Optional[str] = None
    screenshots_base64: Optional[List[str]] = None

    class Config:
        extra = "allow"


class CompareRequest(BaseModel):
    left: CompareSide
    right: CompareSide
    profile: Optional[Profile] = None

    class Config:
        extra = "allow"


# OtoBot için basit soru modeli
class OtoBotRequest(BaseModel):
    question: Optional[str] = None   # Şu an tek soru yeterli
    history: Optional[List[Dict[str, str]]] = None  # İleride sohbetli yaparız

    class Config:
        extra = "allow"


# ---------------------------------------------------------
# Boş istek kontrolü
# ---------------------------------------------------------
def ensure_has_some_content(req: AnalyzeRequest) -> None:
    has_basic_vehicle = bool((req.vehicle.make or "").strip() or (req.vehicle.model or "").strip())
    has_desc = bool(req.ad_description and req.ad_description.strip())

    all_ss: List[str] = []
    if req.screenshot_base64:
        all_ss.append(req.screenshot_base64)
    if req.screenshots_base64:
        all_ss.extend([s for s in req.screenshots_base64 if s])

    has_screenshot = len(all_ss) > 0

    if not (has_basic_vehicle or has_desc or has_screenshot):
        raise HTTPException(
            status_code=400,
            detail="Boş istek. En azından marka/model, ilan açıklaması veya ekran görüntüsü gönder."
        )


# ---------------------------------------------------------
# Backend tarafı: kabaca maliyet & risk tahmini
# (Segment, yaş, km, yakıt vb. ile oynayıp GPT'ye bilgi veriyoruz)
# ---------------------------------------------------------
def guess_segment(vehicle: Vehicle) -> str:
    name = (vehicle.make + " " + vehicle.model).lower()

    b_hatch = ["clio", "fiesta", "yaris", "i20", "polo", "corsa", "fabia"]
    c_sedan = ["corolla", "focus", "megane", "civic", "astra", "egea"]
    c_suv = ["duster", "kuga", "qashqai", "sportage", "tucson", "3008"]
    d_sedan = ["passat", "superb", "508", "insignia"]
    premium = ["bmw", "mercedes", "audi", "volvo", "range rover"]

    if any(k in name for k in b_hatch):
        return "B-segment küçük hatchback"
    if any(k in name for k in c_sedan):
        return "C-segment aile sedan/hatchback"
    if any(k in name for k in c_suv):
        return "C-segment SUV"
    if any(k in name for k in d_sedan):
        return "D-segment konfor sedan"
    if any(k in name for k in premium):
        return "premium D/E-segment"

    return "genel C/D-segment binek araç"


def estimate_costs_and_risks(req: AnalyzeRequest) -> Dict[str, Any]:
    v = req.vehicle
    p = req.profile or Profile()

    segment = guess_segment(v)
    age: Optional[int] = None
    if v.year:
        age = max(0, 2025 - v.year)  # yılı 2025 varsayıyorum, çok kritik değil

    mileage = v.mileage_km or 0

    # Basit bandlar (tam rakamlar önemli değil, GPT'ye yön veriyor)
    base_maintenance = 15000  # TL
    base_fuel = 25000

    # Segment çarpanları
    seg_mult = 1.0
    if "B-segment" in segment:
        seg_mult = 0.7
    elif "C-segment" in segment:
        seg_mult = 1.0
    elif "SUV" in segment:
        seg_mult = 1.3
    elif "premium" in segment or "D-segment" in segment:
        seg_mult = 1.6

    # Yaş ve km etkisi
    age_mult = 1.0
    if age is not None:
        if age > 12:
            age_mult = 1.6
        elif age > 8:
            age_mult = 1.3
        elif age > 5:
            age_mult = 1.1

    km_mult = 1.0
    if mileage > 250_000:
        km_mult = 1.7
    elif mileage > 180_000:
        km_mult = 1.4
    elif mileage > 120_000:
        km_mult = 1.2

    # Yakıt tipi
    fuel_mult = 1.0
    fuel_risk = "orta"
    if v.fuel == "diesel":
        fuel_mult = 1.1
        if mileage > 180_000 and p.usage == "city":
            fuel_risk = "yüksek (DPF / enjektör riski)"
    elif v.fuel == "lpg":
        fuel_mult = 0.9
        fuel_risk = "orta (LPG montaj kalitesine bağlı)"
    elif v.fuel in ("hybrid", "electric"):
        fuel_mult = 0.8
        fuel_risk = "düşük-orta (batarya sağlığına bağlı)"

    yearly_maintenance = int(base_maintenance * seg_mult * age_mult * km_mult)
    yearly_fuel = int(base_fuel * seg_mult * ((p.yearly_km / 15000) or 1) * fuel_mult)

    # Sigorta seviyesi (çok kaba)
    if "premium" in segment:
        insurance_level = "yüksek"
    elif "SUV" in segment or "D-segment" in segment:
        insurance_level = "orta-yüksek"
    else:
        insurance_level = "orta"

    # Resale hızı
    if "C-segment" in segment or "B-segment" in segment:
        resale_speed = "hızlı"
    elif "SUV" in segment:
        resale_speed = "orta-hızlı"
    else:
        resale_speed = "orta"

    # Genel risk (bilgi amaçlı; Keşfet preview'ünde direkt 'riskli' demeyeceğiz)
    risk_level = "orta"
    risk_notes: List[str] = []
    if age is not None and age > 12:
        risk_level = "yüksek"
        risk_notes.append("İleri yaş nedeniyle kronik masraflar artabilir.")
    if mileage > 250_000:
        risk_level = "yüksek"
        risk_notes.append("Km çok yüksek, motor/şanzıman revizyon riski.")
    if "yüksek" in fuel_risk:
        risk_level = "yüksek"

    return {
        "segment_guess": segment,
        "age": age,
        "mileage_km": mileage,
        "estimated_yearly_maintenance_tr": yearly_maintenance,
        "estimated_yearly_fuel_tr": yearly_fuel,
        "insurance_level": insurance_level,
        "resale_speed": resale_speed,
        "fuel_risk_comment": fuel_risk,
        "overall_risk_level": risk_level,
        "risk_notes": risk_notes,
    }


# ---------------------------------------------------------
# Kullanıcı mesajını tek string haline getiriyoruz
# ---------------------------------------------------------
def build_user_content(req: AnalyzeRequest, mode: str) -> str:
    v = req.vehicle
    p = req.profile or Profile()

    ad_text = (req.ad_description or "").strip()

    all_ss: List[str] = []
    if req.screenshot_base64:
        all_ss.append(req.screenshot_base64)
    if req.screenshots_base64:
        all_ss.extend([s for s in req.screenshots_base64 if s])

    backend_context = estimate_costs_and_risks(req)

    ss_info = ""
    if all_ss:
        ss_info = (
            f"\nKullanıcı {len(all_ss)} adet ilan ekran görüntüsü ekledi. "
            "Bu görüntülerdeki fiyat, donanım, paket ve hasar bilgilerini de analizinde kullan. "
            "Eğer görüntülere doğrudan erişemiyorsan bile, bu bilgilerin mevcut olduğunu "
            "varsayarak genel bir değerlendirme yap."
        )

    base_text = f"""
Kullanıcı Oto Analiz uygulamasında **{mode}** modunda analiz istiyor.

Araç bilgileri:
- Marka: {v.make}
- Model: {v.model}
- Yıl: {v.year or "-"}
- Kilometre: {v.mileage_km or "-"} km
- Yakıt: {v.fuel or p.fuel_preference}

Kullanım profili:
- Yıllık km: {p.yearly_km} km
- Kullanım tipi: {p.usage}
- Yakıt tercihi: {p.fuel_preference}
"""

    if ad_text:
        base_text += f"\nİlan açıklaması:\n{ad_text}\n"

    # Backend’in kaba tahmini – model bunu kullanarak daha gerçekçi yorum yapsın
    base_text += "\n--- Backend tahmini maliyet & risk bilgileri (kaba hesap) ---\n"
    base_text += json.dumps(backend_context, ensure_ascii=False)
    base_text += "\n-----------------------------------------------------------\n"
    base_text += ss_info

    return base_text.strip()


# ---------------------------------------------------------
# LLM çağrısı (JSON mode)
# ---------------------------------------------------------
def call_llm_json(model_name: str, system_prompt: str, user_content: str) -> Dict[str, Any]:
    try:
        resp = client.chat.completions.create(
            model=model_name,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LLM isteği başarısız ({model_name}): {e}",
        )

    try:
        content = resp.choices[0].message.content
        if isinstance(content, str):
            return json.loads(content)
        return content  # type: ignore[return-value]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LLM yanıtı JSON parse edilemedi: {e}",
        )


# ---------------------------------------------------------
# System promptlar
# ---------------------------------------------------------
SYSTEM_PROMPT_NORMAL = """
Sen 'Oto Analiz' uygulaması için çalışan bir ARAÇ İLANI ANALİZ ASİSTANI'sın.
Görevin: Kullanıcının verdiği araç bilgilerine, ilan açıklamasına ve backend'in
sağladığı tahmini maliyet/risk bilgilerine göre NET ve ORTA DETAYLI bir analiz yapmak.

ÇIKTIYI SADECE GEÇERLİ BİR JSON OLARAK DÖN:

{
  "scores": {
    "overall_100": sayı,
    "mechanical_100": sayı,
    "body_100": sayı,
    "economy_100": sayı
  },
  "summary": {
    "short_comment": "1-2 cümlelik genel yorum",
    "pros": ["madde madde artılar"],
    "cons": ["madde madde eksiler"],
    "estimated_risk_level": "düşük" | "orta" | "yüksek"
  },
  "preview": {
    "title": "Araç başlığı (marka + model + yıl)",
    "price_tag": "Uygun" | "Normal" | "Yüksek" | null,
    "spoiler": "Keşfet için 1-2 cümlelik kısa ve NÖTR özet. Burada 'alınır, alınmaz, sakın, riskli' gibi kelimeler kullanma.",
    "bullets": [
      "En fazla 3 madde. 'Ekspertiz önerilir', 'tramer kontrolü yapılmalı' gibi nötr, bilgilendirici cümleler yaz."
    ]
  }
}

Kurallar:
- Kullanıcıyı korkutma, kesin hükümler verme. Özellikle PREVIEW alanında:
  - 'alınır', 'alınmaz', 'sakın', 'riskli', 'tehlikeli' gibi kelimeleri KULLANMA.
  - Fiyat ile ilgili sadece genel etiket ver ('Uygun/Normal/Yüksek'), rakam yazma.
- SADECE JSON DÖN, JSON dışında metin yazma.
- Dil: Türkçe.
"""

SYSTEM_PROMPT_PREMIUM = """
Sen 'Oto Analiz' uygulamasının PREMIUM analiz asistanısın.
Normal analizdeki her şeyi yap, ama daha detaylı ve piyasa odaklı anlat.

Türkiye koşullarını varsay:
- Parça bulunabilirliği, servis ağı, kronik sorun riskini değerlendir.
- Yakıt türüne göre tahmini yakıt maliyetini yorumla (backend tahminini de kullan).
- Segmentine göre bakım, kasko ve ikinci el piyasasını anlat (hızlı mı satılır?).

ÇIKTIYI SADECE GEÇERLİ BİR JSON OLARAK DÖN:

{
  "scores": {
    "overall_100": sayı,
    "mechanical_100": sayı,
    "body_100": sayı,
    "economy_100": sayı,
    "comfort_100": sayı,
    "family_use_100": sayı,
    "resale_100": sayı
  },
  "cost_estimates": {
    "yearly_maintenance_tr": sayı,
    "yearly_fuel_tr": sayı,
    "insurance_level": "düşük" | "orta" | "orta-yüksek" | "yüksek",
    "notes": "kısa maliyet özeti"
  },
  "risk_analysis": {
    "chronic_issues": ["olası kronik sorunlar"],
    "risk_level": "düşük" | "orta" | "yüksek",
    "warnings": ["dikkat edilmesi gereken noktalar"]
  },
  "summary": {
    "short_comment": "1-2 cümlelik genel yorum",
    "pros": ["madde madde artılar"],
    "cons": ["madde madde eksiler"],
    "who_should_buy": "Bu araç kimler için mantıklı?"
  },
  "preview": {
    "title": "Araç başlığı (marka + model + yıl)",
    "price_tag": "Uygun" | "Normal" | "Yüksek" | null,
    "spoiler": "Keşfet için 1-2 cümlelik kısa ve NÖTR özet. Burada 'alınır, alınmaz, sakın, riskli' gibi kelimeler kullanma.",
    "bullets": [
      "En fazla 3 madde. 'Ekspertiz önerilir', 'tramer kontrolü yapılmalı' gibi nötr, bilgilendirici cümleler yaz."
    ]
  }
}

Kurallar:
- Kullanıcıyı korkutmadan ama dürüst ol.
- PREVIEW kısmı Keşfet için kullanılacak, bu yüzden:
  - Kesin karar cümleleri veya ağır ifadeler kullanma.
  - Fiyat rakamı yazma, sadece 'Uygun/Normal/Yüksek' etiketi ver.
- SADECE JSON DÖN, JSON dışında metin yazma.
- Dil: Türkçe.
"""

SYSTEM_PROMPT_COMPARE = """
Sen 'Oto Analiz' uygulaması için ARAÇ KARŞILAŞTIRMA asistanısın.
Kullanıcıya iki aracı teknik, maliyet ve kullanım açısından karşılaştır.

Çıktın sadece JSON olsun:
{
  "better_overall": "left" | "right",
  "summary": "kısa genel değerlendirme",
  "left_pros": ["sol araç artıları"],
  "left_cons": ["sol araç eksileri"],
  "right_pros": ["sağ araç artıları"],
  "right_cons": ["sağ araç eksileri"],
  "use_cases": {
    "family_use": "hangi araç daha mantıklı ve neden",
    "long_distance": "...",
    "city_use": "..."
  }
}
Dil: Türkçe, sadece JSON.
"""

SYSTEM_PROMPT_OTOBOT = """
Sen 'Oto Analiz' uygulamasının OTOBOT isimli araç alma rehberisin.
Kullanıcı bütçesini, kullanım şeklini ve beklentisini anlatıyor olabilir.
Görevin: Ona Türkiye piyasasına göre mantıklı segment ve model önerileri vermek.

Çıktı sadece JSON olsun:
{
  "answer": "kullanıcıya verilen detaylı tavsiye yanıtı (Türkçe)",
  "suggested_segments": ["B-SUV", "C-sedan", ...],
  "example_models": ["Örnek model 1", "Örnek model 2", "..."]
}
Dil: Türkçe, dışarıda metin yok.
"""


# ---------------------------------------------------------
# Healthcheck
# ---------------------------------------------------------
@app.get("/")
async def root() -> Dict[str, Any]:
    return {"ok": True, "message": "Oto Analiz backend çalışıyor."}


# ---------------------------------------------------------
# NORMAL ANALİZ
# ---------------------------------------------------------
@app.post("/analyze")
async def analyze(req: AnalyzeRequest) -> Dict[str, Any]:
    ensure_has_some_content(req)
    user_content = build_user_content(req, mode="normal")

    data = call_llm_json(
        model_name=OPENAI_MODEL_NORMAL,
        system_prompt=SYSTEM_PROMPT_NORMAL,
        user_content=user_content,
    )
    return data


# ---------------------------------------------------------
# PREMIUM ANALİZ
# ---------------------------------------------------------
@app.post("/premium_analyze")
async def premium_analyze(req: AnalyzeRequest) -> Dict[str, Any]:
    ensure_has_some_content(req)
    user_content = build_user_content(req, mode="premium")

    data = call_llm_json(
        model_name=OPENAI_MODEL_PREMIUM,
        system_prompt=SYSTEM_PROMPT_PREMIUM,
        user_content=user_content,
    )
    return data


# ---------------------------------------------------------
# ARAÇ KARŞILAŞTIRMA
# ---------------------------------------------------------
@app.post("/compare_analyze")
async def compare_analyze(req: CompareRequest) -> Dict[str, Any]:
    left_v = req.left.vehicle
    right_v = req.right.vehicle

    left_text = f"""
Sol araç:
Marka: {left_v.make}, Model: {left_v.model}, Yıl: {left_v.year}, Km: {left_v.mileage_km}, Yakıt: {left_v.fuel}
İlan açıklaması: {req.left.ad_description or "-"}
"""

    right_text = f"""
Sağ araç:
Marka: {right_v.make}, Model: {right_v.model}, Yıl: {right_v.year}, Km: {right_v.mileage_km}, Yakıt: {right_v.fuel}
İlan açıklaması: {req.right.ad_description or "-"}
"""

    profile_text = ""
    if req.profile:
        p = req.profile
        profile_text = f"""
Kullanıcı profili:
- Yıllık km: {p.yearly_km}
- Kullanım: {p.usage}
- Yakıt tercihi: {p.fuel_preference}
"""

    user_content = (left_text + "\n" + right_text + "\n" + profile_text).strip()

    data = call_llm_json(
        model_name=OPENAI_MODEL_COMPARE,
        system_prompt=SYSTEM_PROMPT_COMPARE,
        user_content=user_content,
    )
    return data


# ---------------------------------------------------------
# OTOBOT – Araç alma rehberi
# ---------------------------------------------------------
@app.post("/otobot")
async def otobot(req: OtoBotRequest) -> Dict[str, Any]:
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(
            status_code=400,
            detail="Soru boş olamaz. 'question' alanına bir metin gönder.",
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_OTOBOT},
        {"role": "user", "content": question},
    ]

    # İleride history gelirse ekle
    if req.history:
        for h in req.history:
            role = h.get("role", "user")
            content = h.get("content", "")
            if content:
                messages.append({"role": role, "content": content})

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL_OTOBOT,
            response_format={"type": "json_object"},
            messages=messages,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LLM isteği başarısız (OtoBot): {e}",
        )

    try:
        content = resp.choices[0].message.content
        if isinstance(content, str):
            return json.loads(content)
        return content  # type: ignore[return-value]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"OtoBot yanıtı JSON parse edilemedi: {e}",
        )
