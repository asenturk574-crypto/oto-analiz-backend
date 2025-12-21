import os
import json
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing_extensions import Literal
from dotenv import load_dotenv
from openai import OpenAI

# -------------------------------------------------------------------
#  ENV & OPENAI CLIENT
# -------------------------------------------------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY bulunamadı. .env dosyanı kontrol et.")

client = OpenAI(api_key=api_key)

# Varsayılan modeller (istersen .env'den değiştirebilirsin)
DEFAULT_MODEL_NORMAL = os.getenv("OPENAI_MODEL_NORMAL", "gpt-4.1-mini")
DEFAULT_MODEL_PREMIUM = os.getenv("OPENAI_MODEL_PREMIUM", "gpt-4.1")
DEFAULT_MODEL_OTOBOT = os.getenv("OPENAI_MODEL_OTOBOT", "gpt-4.1-mini")

# -------------------------------------------------------------------
#  FASTAPI APP
# -------------------------------------------------------------------
app = FastAPI(
    title="Oto Analiz Backend",
    description="Oto Analiz mobil uygulaması için normal/premium analiz API'si",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # İstersek ileride domain ile kısıtlarız
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
#  MODELLER
# -------------------------------------------------------------------


class Profile(BaseModel):
    yearly_km: int = Field(15000, ge=0)
    usage: Literal["city", "mixed", "highway"] = "mixed"
    fuel_preference: Literal["gasoline", "diesel", "lpg", "hybrid", "electric"] = "diesel"


class Vehicle(BaseModel):
    make: Optional[str] = None
    model: Optional[str] = None
    year: Optional[int] = None
    mileage_km: Optional[int] = None
    fuel: Optional[str] = None
    transmission: Optional[str] = None
    body_type: Optional[str] = None  # sedan, hatchback, suv...
    segment: Optional[str] = None  # B, C, D...
    price: Optional[float] = None
    city: Optional[str] = None


class AnalyzeRequest(BaseModel):
    # Hangi ekran: normal / premium / manual / compare / otobot
    mode: Literal["normal", "premium", "manual", "compare", "otobot"] = "normal"

    # Serbest metin alanları
    text: Optional[str] = None  # genel açıklama
    listing_title: Optional[str] = None
    listing_description: Optional[str] = None
    ad_description: Optional[str] = None  # eski frontend ile uyum için

    # SS alanları
    screenshot_base64: Optional[str] = None  # tek görsel (eski sürüm)
    screenshots_base64: Optional[List[str]] = None  # birden fazla görsel

    # Yapısal veriler
    profile: Optional[Profile] = None
    vehicle: Optional[Vehicle] = None

    # Gelecekte ek bilgi için
    context: Optional[Dict[str, Any]] = None


# -------------------------------------------------------------------
#  SABİT VERİ TABLOLARI (bakım, lastik, sigorta, likidite vb.)
#  Bunlar tamamen backend içinde; dışarı veri sızdırmıyor.
# -------------------------------------------------------------------

SEGMENT_COSTS = {
    # Ortalama yıllık bakım maliyeti (TL) + tahmini lastik set fiyatı (TL)
    "B": {"maintenance_yearly": 12000, "tire_set": 9000},
    "C": {"maintenance_yearly": 15000, "tire_set": 11000},
    "D": {"maintenance_yearly": 20000, "tire_set": 14000},
    "E": {"maintenance_yearly": 26000, "tire_set": 17000},
    "SUV-C": {"maintenance_yearly": 22000, "tire_set": 16000},
    "SUV-D": {"maintenance_yearly": 28000, "tire_set": 19000},
}

INSURANCE_LOSS_RATIO = {
    # Sigortacının gözünden: hasar prim oranı yüksekse kasko pahalı olur
    "low": 0.8,
    "medium": 1.0,
    "high": 1.25,
}

BRAND_RISK_NOTES = {
    # Çok özet “marka + şanzıman” / “marka + motor” risk notları – sadece örnek
    "vw_dsg": "DSG şanzımanlı VW modellerinde geçmişte kavrama ve mekatronik kaynaklı masraf riskleri görülmüştür.",
    "renault_1.5_dci": "1.5 dCi motorlar genel olarak ekonomik; fakat bakımsız örneklerde turbo/enjektör masrafı çıkabiliyor.",
    "bmw_n47": "Eski N47 dizel BMW motorlarında zamanlama zinciri masraf riski biliniyor, kronik olarak takip edilmeli.",
    "vag_tsi_early": "Erken nesil TSI motorlarda yağ tüketimi ve karbon birikimi daha sık raporlanmış durumda.",
}

LIQUIDITY_SCORES = {
    # 1–5: 5 = piyasası çok hızlı, 1 = zor satılır
    "corolla": 5,
    "civic": 5,
    "egea": 4,
    "megane": 4,
    "passat": 4,
    "focus": 3,
    "premium_suv": 3,
}

PARTS_AVAILABILITY = {
    # 1–5: 5 = her yerde parça / usta var
    "renault": 5,
    "fiat": 5,
    "toyota": 4,
    "honda": 4,
    "vw": 4,
    "bmw": 3,
    "mercedes": 3,
    "range_rover": 2,
}


# -------------------------------------------------------------------
#  YARDIMCI FONKSİYONLAR
# -------------------------------------------------------------------


def _ensure_has_some_text(req: AnalyzeRequest) -> None:
    """Boş istek gelmesini engelle."""
    if any(
        [
            req.text and req.text.strip(),
            req.listing_title and req.listing_title.strip(),
            req.listing_description and req.listing_description.strip(),
            req.ad_description and req.ad_description.strip(),
        ]
    ):
        return

    raise HTTPException(
        status_code=400,
        detail="Boş istek. En azından 'text' veya 'listing_title/description' ya da 'ad_description' gönder.",
    )


def _guess_segment(v: Optional[Vehicle]) -> str:
    if not v:
        return "C"
    # Kullanıcının segment girmesi durumunda onu kullan
    if v.segment:
        return v.segment.upper()

    make = (v.make or "").lower()
    model = (v.model or "").lower()
    body = (v.body_type or "").lower()

    # Çok kaba tahminler – sadece bağlam için
    if any(x in model for x in ["corolla", "civic", "megane", "focus", "astra"]):
        return "C"
    if any(x in model for x in ["passat", "superb", "accord", "camry"]):
        return "D"
    if "egea" in model:
        return "C"
    if "sportage" in model or "tucson" in model or "qashqai" in model:
        return "SUV-C"
    if "x5" in model or "gle" in model or "range" in make or "range" in model:
        return "SUV-D"

    if "suv" in body:
        return "SUV-C"
    if "hb" in body or "hatch" in body:
        return "B"

    return "C"


def build_structured_context(req: AnalyzeRequest) -> Dict[str, Any]:
    """Segment, bakım, yakıt, sigorta vb. için sayısal bağlam üretir."""
    v = req.vehicle
    p = req.profile

    seg = _guess_segment(v)
    seg_cost = SEGMENT_COSTS.get(seg, SEGMENT_COSTS["C"])

    yearly_km = p.yearly_km if p else 15000
    usage = p.usage if p else "mixed"
    fuel_pref = p.fuel_preference if p else (v.fuel if v and v.fuel else "diesel")

    # Yakıt tüketim tahmini (çok kaba)
    base_l_100 = {
        "gasoline": 7.5,
        "diesel": 6.0,
        "lpg": 9.0,
        "hybrid": 5.5,
        "electric": 0.0,
    }.get(fuel_pref, 7.0)

    usage_coef = {"city": 1.2, "mixed": 1.0, "highway": 0.85}.get(usage, 1.0)
    est_l_100 = base_l_100 * usage_coef

    # Yıllık yakıt masrafı için kabaca 40 TL / litre varsayalım
    est_fuel_cost_year = yearly_km / 100 * est_l_100 * 40

    # Sigorta/kasko kabaca segment + risk notu
    risk_key = "medium"
    brand = (v.make or "").lower()
    model = (v.model or "").lower()

    if "bmw" in brand or "mercedes" in brand or "audi" in brand:
        risk_key = "high"
    elif "fiat" in brand or "renault" in brand or "hyundai" in brand:
        risk_key = "low"

    risk_coef = INSURANCE_LOSS_RATIO.get(risk_key, 1.0)
    base_kasko = {
        "B": 12000,
        "C": 15000,
        "D": 20000,
        "E": 26000,
        "SUV-C": 22000,
        "SUV-D": 28000,
    }.get(seg, 15000)

    est_kasko = int(base_kasko * risk_coef)

    # Likidite (piyasa hızı)
    liq_key = None
    if "corolla" in model:
        liq_key = "corolla"
    elif "civic" in model:
        liq_key = "civic"
    elif "megane" in model:
        liq_key = "megane"
    elif "egea" in model:
        liq_key = "egea"
    elif "passat" in model:
        liq_key = "passat"
    elif any(x in model for x in ["x5", "gle", "q7", "range"]):
        liq_key = "premium_suv"

    liquidity = LIQUIDITY_SCORES.get(liq_key or "", 3)

    # Parça bulunabilirliği
    part_key = None
    if brand in ["renault", "dacia"]:
        part_key = "renault"
    elif brand in ["fiat", "tofaş"]:
        part_key = "fiat"
    elif brand in ["toyota"]:
        part_key = "toyota"
    elif brand in ["honda"]:
        part_key = "honda"
    elif brand in ["volkswagen", "vw"]:
        part_key = "vw"
    elif brand in ["bmw"]:
        part_key = "bmw"
    elif brand in ["mercedes", "mercedes-benz"]:
        part_key = "mercedes"
    elif "range" in brand or "land rover" in brand:
        part_key = "range_rover"

    parts_score = PARTS_AVAILABILITY.get(part_key or "", 3)

    # Marka bazlı risk notu
    brand_risks: List[str] = []
    if "vw" in brand and "dsg" in (req.text or "").lower():
        brand_risks.append(BRAND_RISK_NOTES["vw_dsg"])
    if "dci" in (req.text or "").lower() and "renault" in brand:
        brand_risks.append(BRAND_RISK_NOTES["renault_1.5_dci"])
    if "n47" in (req.text or "").lower() and "bmw" in brand:
        brand_risks.append(BRAND_RISK_NOTES["bmw_n47"])

    return {
        "segment": seg,
        "maintenance_yearly_est_tl": seg_cost["maintenance_yearly"],
        "tire_set_est_tl": seg_cost["tire_set"],
        "fuel_type": fuel_pref,
        "usage": usage,
        "yearly_km": yearly_km,
        "fuel_l_per_100km_est": round(est_l_100, 1),
        "fuel_cost_year_est_tl": int(est_fuel_cost_year),
        "kasko_est_tl": est_kasko,
        "liquidity_score_1_5": liquidity,
        "parts_availability_1_5": parts_score,
        "brand_risks": brand_risks,
    }


def build_user_content(req: AnalyzeRequest, ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    """LLM'e gidecek vision + text content (multi-image dahil)."""
    parts: List[Dict[str, Any]] = []

    # 1) Metinler
    text_chunks: List[str] = []

    if req.listing_title:
        text_chunks.append(f"İlan başlığı: {req.listing_title.strip()}")

    combined_desc = ""
    for t in [
        req.text,
        req.listing_description,
        req.ad_description,
    ]:
        if t and t.strip():
            combined_desc += t.strip() + "\n"

    if combined_desc:
        text_chunks.append("İlan açıklaması / ek notlar:\n" + combined_desc.strip())

    if req.vehicle:
        v = req.vehicle
        text_chunks.append(
            "Araç temel verileri: "
            f"marka={v.make}, model={v.model}, yıl={v.year}, km={v.mileage_km}, "
            f"yakıt={v.fuel}, vites={v.transmission}, kasa={v.body_type}, segment={v.segment}, fiyat={v.price}"
        )

    if ctx:
        text_chunks.append(
            "Yapısal bağlam JSON (bakım, yakıt, sigorta, likidite vb.):\n"
            + json.dumps(ctx, ensure_ascii=False)
        )

    if text_chunks:
        parts.append({"type": "text", "text": "\n\n".join(text_chunks)})

    # 2) Görseller (tek + çoklu birlikte)
    images: List[str] = []
    if req.screenshot_base64:
        images.append(req.screenshot_base64)
    if req.screenshots_base64:
        images.extend([b for b in req.screenshots_base64 if b])

    for b64 in images:
        parts.append(
            {
                "type": "input_image",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                },
            }
        )

    if not parts:
        parts.append(
            {
                "type": "text",
                "text": "Kullanıcı hiçbir metin ya da görsel göndermedi. En azından genel bir ikinci el piyasa bilgisi ver.",
            }
        )

    return parts


def call_llm(model_name: str, system_prompt: str, req: AnalyzeRequest) -> Dict[str, Any]:
    """
    OpenAI Responses API çağrısı.
    Dikkat: Burada 'response_format' KULLANMIYORUZ; bu yüzden 500 hatası çözülecek.
    JSON'u prompt ile istiyoruz ve sonra kendimiz parse ediyoruz.
    """
    ctx = build_structured_context(req)
    user_content = build_user_content(req, ctx)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    try:
        resp = client.responses.create(
            model=model_name,
            input=messages,
            temperature=0.4,
        )
        # text parçalarını topla
        output = resp.output[0].content  # type: ignore[attr-defined]
        text_chunks = []
        for c in output:
            if getattr(c, "type", None) == "output_text":
                text_chunks.append(c.text)

        raw = "".join(text_chunks).strip()

        # Model düzgün JSON vermezse, fallback bir yapı döndürelim
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {
                "raw_text": raw,
                "scores": {},
                "summary": {"pros": [], "cons": []},
            }

        # Zorunlu alanları garanti altına al
        data.setdefault("scores", {})
        data.setdefault("summary", {})
        data["summary"].setdefault("pros", [])
        data["summary"].setdefault("cons", [])

        return data

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LLM isteği başarısız ({model_name}): {e}",
        )


def build_system_prompt_for_mode(mode: str) -> str:
    """
    Farklı ekranlar için farklı rol talimatı.
    Şu an çok detaylı yazmak yerine özet veriyoruz, ama sektör,
    kronik sorun, masraf ve pazarlık odağı hepsinde var.
    """
    base = (
        "Sen Oto Analiz uygulaması için uzman bir araç ilanı analiz asistanısın. "
        "Kullanıcının ilan metni, araç bilgileri ve varsa ekran görüntülerinden "
        "yararlanarak Türkiye ikinci el piyasasına göre dürüst, tarafsız ve net yorum yap. "
        "Kronik sorun riskleri, muhtemel masraflar, artı/eksi yönler, yıllık maliyet ve "
        "pazarlık tavsiyesi mutlaka olsun. Cevabını her zaman JSON formatında ver."
    )

    if mode == "premium":
        return (
            base
            + "\nAyrıca premium moddasın: bakım maliyeti, yakıt masrafı, sigorta/kasko, "
            "likidite (piyasada ne kadar hızlı satılır), parça bulunabilirliği ve marka-risk "
            "notlarını daha detaylı yorumla. Kullanıcının profilini (yıllık km, kullanım tipi, "
            "yakıt tercihi) mutlaka dikkate al."
        )
    if mode == "manual":
        return (
            base
            + "\nBu modda kullanıcı kendi aracını manuel giriyor. Özellikle uzun vadeli sahiplik, "
            "bakım planı ve olası büyük masraflara odaklan."
        )
    if mode == "compare":
        return (
            base
            + "\nBu modda en az iki aracı karşılaştırıyorsun. Hangisi daha mantıklı, hangi profilde "
            "hangi aracı önerdiğini açıkça yaz ve JSON içinde her araç için ayrı puanlar ver."
        )
    if mode == "otobot":
        return (
            base
            + "\nBu modda 'Hangi aracı almalıyım?' asistanısın. Kullanıcının bütçesine, profil "
            "bilgilerine göre segment/örnek model öner ve JSON içinde öneri listesi üret."
        )
    # normal
    return base


# -------------------------------------------------------------------
#  ENDPOINTLER
# -------------------------------------------------------------------


@app.get("/")
async def root() -> Dict[str, Any]:
    return {"status": "ok", "message": "Oto Analiz backend çalışıyor."}


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "healthy"}


@app.post("/analyze")
async def analyze(req: AnalyzeRequest) -> Dict[str, Any]:
    _ensure_has_some_text(req)

    mode = req.mode or "normal"
    system_prompt = build_system_prompt_for_mode(mode)

    if mode == "premium":
        model_name = DEFAULT_MODEL_PREMIUM
    elif mode == "otobot":
        model_name = DEFAULT_MODEL_OTOBOT
    else:
        model_name = DEFAULT_MODEL_NORMAL

    data = call_llm(model_name, system_prompt, req)

    # Frontend için ortak formatta döndüğümüzden emin olalım
    return {
        "mode": mode,
        "scores": data.get("scores", {}),
        "summary": data.get("summary", {}),
        "raw": data,  # istersen Flutter tarafında debug için kullanırsın
    }
