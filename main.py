# main.py — Oto Analiz Backend (Güncel Premium + Masraf + Piyasa + Uygunluk + Sigorta/Kasko + Likidite + Kronik)
# Notlar:
# - Bu dosya mevcut yapını BOZMADAN premium çıktıyı daha uzun/detaylı hale getirir.
# - “Piyasa fiyatı” tarafında web/veri çekmeden %100 kesin rakam verilemez; bu yüzden aralık + güven skoru + gerekçe üretir.
# - Premium analiz “kısa gelirse” otomatik ikinci çağrı ile detay tamamlatır (detay guard).

import os
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------
# Ortam değişkenleri ve OpenAI client
# ---------------------------------------------------------
load_dotenv()

CURRENT_YEAR = int(os.getenv("CURRENT_YEAR", "2025"))

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("UYARI: OPENAI_API_KEY bulunamadı, analizler fallback modunda çalışacak.")
    client = None
else:
    client = OpenAI(api_key=api_key)

OPENAI_MODEL_DEFAULT = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_MODEL_NORMAL = os.getenv("OPENAI_MODEL_NORMAL", OPENAI_MODEL_DEFAULT)
OPENAI_MODEL_PREMIUM = os.getenv("OPENAI_MODEL_PREMIUM", OPENAI_MODEL_DEFAULT)
OPENAI_MODEL_COMPARE = os.getenv("OPENAI_MODEL_COMPARE", OPENAI_MODEL_DEFAULT)
OPENAI_MODEL_OTOBOT = os.getenv("OPENAI_MODEL_OTOBOT", OPENAI_MODEL_DEFAULT)

# Token/çıktı ayarları (premium daha uzun)
MAX_TOKENS_NORMAL = int(os.getenv("MAX_TOKENS_NORMAL", "900"))
MAX_TOKENS_PREMIUM = int(os.getenv("MAX_TOKENS_PREMIUM", "1700"))
MAX_TOKENS_COMPARE = int(os.getenv("MAX_TOKENS_COMPARE", "900"))
MAX_TOKENS_OTOBOT = int(os.getenv("MAX_TOKENS_OTOBOT", "700"))

TEMP_NORMAL = float(os.getenv("TEMP_NORMAL", "0.6"))
TEMP_PREMIUM = float(os.getenv("TEMP_PREMIUM", "0.7"))
TEMP_COMPARE = float(os.getenv("TEMP_COMPARE", "0.6"))
TEMP_OTOBOT = float(os.getenv("TEMP_OTOBOT", "0.6"))

# ---------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------
app = FastAPI(title="Oto Analiz Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# Modeller
# ---------------------------------------------------------
class Profile(BaseModel):
    yearly_km: int = Field(15000, ge=0, le=100000)
    usage: str = "mixed"              # city / mixed / highway
    fuel_preference: str = "gasoline" # gasoline / diesel / lpg / hybrid / electric

    # opsiyonel (frontend isterse gönderebilir)
    budget_try: Optional[int] = None
    priorities: Optional[List[str]] = None     # ["low_cost","family","comfort","performance","resale","safety"]
    first_car: Optional[bool] = None
    family: Optional[bool] = None
    kids: Optional[bool] = None

    class Config:
        extra = "allow"


class Vehicle(BaseModel):
    make: str = ""
    model: str = ""
    trim: Optional[str] = None
    year: Optional[int] = Field(None, ge=1980, le=2035)
    mileage_km: Optional[int] = Field(None, ge=0)
    fuel: Optional[str] = None                # gasoline / diesel / lpg / hybrid / electric
    transmission: Optional[str] = None        # manual / automatic / cvt / dsg / etc
    body_type: Optional[str] = None           # sedan / hatch / suv / etc
    engine: Optional[str] = None              # 1.5 TSI gibi
    power_hp: Optional[int] = None

    class Config:
        extra = "allow"


class AnalyzeRequest(BaseModel):
    profile: Profile = Field(default_factory=Profile)
    vehicle: Vehicle = Field(default_factory=Vehicle)

    screenshot_base64: Optional[str] = None
    screenshots_base64: Optional[List[str]] = None

    ad_description: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "allow"


class CompareSide(BaseModel):
    title: Optional[str] = None
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


class OtoBotRequest(BaseModel):
    question: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = None

    class Config:
        extra = "allow"


# ---------------------------------------------------------
# Helpers - normalize / parse
# ---------------------------------------------------------
def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("ı", "i").replace("ğ", "g").replace("ş", "s").replace("ö", "o").replace("ü", "u").replace("ç", "c")
    s = re.sub(r"\s+", " ", s)
    return s


def _digits(s: str) -> Optional[int]:
    if not s:
        return None
    d = re.sub(r"[^\d]", "", s)
    if not d:
        return None
    try:
        return int(d)
    except:
        return None


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        if isinstance(x, bool):
            return None
        if isinstance(x, int):
            return x
        if isinstance(x, float):
            return int(x)
        if isinstance(x, str):
            return _digits(x)
        return None
    except:
        return None


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _parse_listed_price(req: AnalyzeRequest) -> Optional[int]:
    # Flutter context: listed_price_text
    try:
        txt = (req.context or {}).get("listed_price_text") or ""
        v = _digits(str(txt))
        if v and v > 10000:
            return v
    except:
        pass

    # Alternatif: listed_price_try direkt gelirse
    try:
        v3 = _safe_int((req.context or {}).get("listed_price_try"))
        if v3 and v3 > 10000:
            return v3
    except:
        pass

    # Bazı denemelerde "text" alanından da fiyat gelebilir
    try:
        txt2 = getattr(req, "text", None)
        if isinstance(txt2, str):
            v2 = _digits(txt2)
            if v2 and v2 > 10000:
                return v2
    except:
        pass
    return None


def _bool_from_text(text: str, keywords: List[str]) -> bool:
    t = _norm(text)
    return any(k in t for k in keywords)


# ---------------------------------------------------------
# SEGMENT PROFİLLERİ (TR piyasası için anchor / band)
# ---------------------------------------------------------
SEGMENT_PROFILES: Dict[str, Dict[str, Any]] = {
    "B_HATCH": {
        "name": "B segment (küçük hatchback)",
        "maintenance_yearly_range": (12000, 25000),
        "insurance_level": "orta",
        "parts": "çok kolay",
        "resale": "hızlı",
        "notes": [
            "Parça ve usta erişimi çok kolay, şehir içi kullanım için ideal.",
            "Doğru bakım yapıldığında yıllık masraflar genelde kontrol edilebilir seviyede kalır."
        ],
    },
    "C_SEDAN": {
        "name": "C segment (aile sedan/hatchback)",
        "maintenance_yearly_range": (15000, 32000),
        "insurance_level": "orta",
        "parts": "kolay",
        "resale": "hızlı-orta",
        "notes": [
            "Türkiye’de en likit segmentlerden biri, ikinci elde alıcı bulmak görece kolay.",
            "B segmente göre konfor ve donanım artarken bakım maliyeti bir miktar yükselir."
        ],
    },
    "C_SUV": {
        "name": "C segment SUV",
        "maintenance_yearly_range": (18000, 38000),
        "insurance_level": "orta-yüksek",
        "parts": "orta-kolay",
        "resale": "orta-hızlı",
        "notes": [
            "SUV gövde nedeniyle lastik, fren ve alt takım maliyeti kompakt araçlara göre yükselebilir.",
            "Aile ve uzun yol kullanımı için dengeli fakat tüketim tarafı biraz daha yüksektir."
        ],
    },
    "D_SEDAN": {
        "name": "D segment (konfor sedan)",
        "maintenance_yearly_range": (22000, 50000),
        "insurance_level": "orta-yüksek",
        "parts": "orta",
        "resale": "orta",
        "notes": [
            "Konfor ve donanım seviyesi yüksek, ancak parça ve bakım maliyetleri C segmente göre belirgin şekilde artar."
        ],
    },
    "PREMIUM_D": {
        "name": "Premium D segment",
        "maintenance_yearly_range": (32000, 75000),
        "insurance_level": "yüksek",
        "parts": "orta-zor",
        "resale": "orta",
        "notes": [
            "Premium sınıfta işçilik, bakım ve parça maliyeti ciddi seviyelere çıkabilir.",
            "Kasko ve sigorta primleri kullanıcı yaşı ve hasar geçmişine göre agresif değişebilir."
        ],
    },
    "E_SEGMENT": {
        "name": "E segment / üst sınıf",
        "maintenance_yearly_range": (45000, 120000),
        "insurance_level": "çok yüksek",
        "parts": "zor",
        "resale": "yavaş-orta",
        "notes": [
            "Üst sınıf büyük gövdeli araçlarda tüm masraf kalemleri bariz şekilde yüksektir.",
        ],
    },
}

# ---------------------------------------------------------
# ANCHOR MODEL VERİTABANI (emsal + kronik)
# ---------------------------------------------------------
ANCHORS: List[Dict[str, Any]] = [
    # B segment
    {"key": "renault clio", "segment": "B_HATCH",
     "aliases": ["clio", "clio 4", "clio 5"],
     "chronic": ["Ön takım burç/salıncak yıpranması (km+yaşla)", "Dizel versiyonlarda EGR/DPF hassasiyeti"],
     "parts": "çok kolay", "resale": "hızlı"},
    {"key": "vw polo", "segment": "B_HATCH",
     "aliases": ["polo"],
     "chronic": ["DSG şanzımanlı versiyonlarda yağ/bakım geçmişi kritik"],
     "parts": "kolay", "resale": "hızlı-orta"},
    {"key": "hyundai i20", "segment": "B_HATCH",
     "aliases": ["i20"],
     "chronic": ["Genelde düşük kronik, bakımı ihmal edilmemeli"],
     "parts": "kolay", "resale": "orta-hızlı"},

    # C sedan / hatch
    {"key": "toyota corolla", "segment": "C_SEDAN",
     "aliases": ["corolla"],
     "chronic": ["Genelde düşük kronik, düzenli bakım en kritik konu"],
     "parts": "kolay", "resale": "hızlı"},
    {"key": "honda civic", "segment": "C_SEDAN",
     "aliases": ["civic"],
     "chronic": ["LPG takılı Civiclerde subap ve ayar takibi önemli"],
     "parts": "kolay", "resale": "hızlı"},
    {"key": "renault megane", "segment": "C_SEDAN",
     "aliases": ["megane"],
     "chronic": ["1.5 dCi motorlarda DPF/EGR ve enjektör hassasiyeti", "EDC şanzımanlılarda yağ/bakım geçmişi önemli"],
     "parts": "kolay", "resale": "hızlı-orta"},
    {"key": "fiat egea", "segment": "C_SEDAN",
     "aliases": ["egea", "tipo"],
     "chronic": ["Multijet dizelde DPF, şehir içi kullanımda is/kurum birikimi", "Trim sesleri kullanıcıya göre değişebilir"],
     "parts": "çok kolay", "resale": "hızlı"},

    # D sedan
    {"key": "vw passat", "segment": "D_SEDAN",
     "aliases": ["passat"],
     "chronic": ["DSG şanzıman geçmişi kritik", "TDI ünitelerde EGR/DPF riski"],
     "parts": "orta", "resale": "orta-hızlı"},
    {"key": "skoda superb", "segment": "D_SEDAN",
     "aliases": ["superb"],
     "chronic": ["DSG bakımı ihmal edilmemeli"],
     "parts": "orta", "resale": "orta"},

    # C SUV
    {"key": "nissan qashqai", "segment": "C_SUV",
     "aliases": ["qashqai"],
     "chronic": ["CVT şanzımanlılarda yağ/bakım geçmişi kritik"],
     "parts": "orta", "resale": "orta-hızlı"},
    {"key": "kia sportage", "segment": "C_SUV",
     "aliases": ["sportage"],
     "chronic": ["Dizel versiyonlarda DPF/EGR riski"],
     "parts": "orta", "resale": "orta"},
    {"key": "hyundai tucson", "segment": "C_SUV",
     "aliases": ["tucson"],
     "chronic": ["Dizel versiyonlarda DPF/EGR riski"],
     "parts": "orta", "resale": "orta"},

    # Premium D
    {"key": "bmw 320", "segment": "PREMIUM_D",
     "aliases": ["320i", "320d", "3.20", "3.20i", "3.20d", "3 series", "3 serisi"],
     "chronic": ["Burç/ön takım zamanla yıpranabilir", "Turbo-dizelde EGR/DPF ve zincir/yağ bakımı ihmal edilmemeli"],
     "parts": "orta-zor", "resale": "orta"},
    {"key": "mercedes c200", "segment": "PREMIUM_D",
     "aliases": ["c180", "c200", "c220", "c class", "c serisi"],
     "chronic": ["Elektronik/konfor donanımlarında yaşa bağlı arızalar", "Bakım standardı kritik"],
     "parts": "orta", "resale": "orta"},
    {"key": "audi a4", "segment": "PREMIUM_D",
     "aliases": ["a4", "a4 allroad", "allroad"],
     "chronic": ["S-tronic/DSG geçmişi önemli", "Baz TFSI nesillerinde yağ tüketimi geçmişi sorgulanmalı"],
     "parts": "orta-zor", "resale": "orta"},
    {"key": "volvo s60", "segment": "PREMIUM_D",
     "aliases": ["s60"],
     "chronic": ["Parça ve servis maliyeti premium seviyede"],
     "parts": "zor", "resale": "orta-yavaş"},
]

# ---------------------------------------------------------
# Segment tespiti
# ---------------------------------------------------------
def detect_segment(make: str, model: str) -> str:
    s = _norm(f"{make} {model}")

    if any(k in s for k in ["bmw", "mercedes", "audi", "volvo", "lexus", "range rover"]):
        return "PREMIUM_D"

    if any(k in s for k in ["clio", "polo", "i20", "corsa", "yaris", "fiesta", "fabia"]):
        return "B_HATCH"
    if any(k in s for k in ["corolla", "civic", "megane", "astra", "focus", "egea", "tipo", "elantra", "i30"]):
        return "C_SEDAN"
    if any(k in s for k in ["qashqai", "tucson", "sportage", "kuga", "3008", "duster"]):
        return "C_SUV"
    if any(k in s for k in ["passat", "superb", "508", "insignia"]):
        return "D_SEDAN"

    return "C_SEDAN"


def find_anchor_matches(make: str, model: str, segment: str, limit: int = 3) -> List[Dict[str, Any]]:
    target = _norm(f"{make} {model}")
    scored: List[Tuple[int, Dict[str, Any]]] = []

    for a in ANCHORS:
        score = 0
        if a.get("segment") == segment:
            score += 5
        aliases = a.get("aliases") or []
        key = _norm(a.get("key", ""))
        if key and key in target:
            score += 10
        for al in aliases:
            if _norm(al) in target:
                score += 8

        if any(m in target for m in ["bmw", "mercedes", "audi", "volvo"]) and \
           any(m in key for m in ["bmw", "mercedes", "audi", "volvo"]):
            score += 2

        if score > 0:
            scored.append((score, a))

    scored.sort(key=lambda x: x[0], reverse=True)

    if scored:
        return [x[1] for x in scored[:limit]]

    return [a for a in ANCHORS if a.get("segment") == segment][:limit]


# ---------------------------------------------------------
# Bilgi kalitesi (yeterince veri var mı?)
# ---------------------------------------------------------
def evaluate_info_quality(req: AnalyzeRequest) -> Dict[str, Any]:
    v = req.vehicle
    p = req.profile or Profile()
    missing: List[str] = []

    if not (v.make and v.make.strip()):
        missing.append("marka")
    if not (v.model and v.model.strip()):
        missing.append("model")
    if v.year is None:
        missing.append("yil")
    if v.mileage_km is None:
        missing.append("km")
    if not v.fuel:
        missing.append("yakit")

    ad_text = (req.ad_description or "").strip()
    ad_len = len(ad_text)
    if ad_len < 40:
        missing.append("ilan_aciklamasi_kisa")

    if not p.yearly_km:
        missing.append("profil_yillik_km")
    if p.usage not in ("city", "mixed", "highway"):
        missing.append("profil_kullanim_tipi")

    # ekran görüntüsü var mı (OCR yok ama varlığı bile bilgi kalitesini artırır çünkü kullanıcı muhtemelen detay ekledi)
    has_ss = bool(req.screenshot_base64) or bool(req.screenshots_base64)

    if len(missing) <= 1 and ad_len >= 120:
        level = "yüksek"
    elif len(missing) <= 3 and (ad_len >= 40 or has_ss):
        level = "orta"
    else:
        level = "düşük"

    return {
        "level": level,
        "missing_fields": missing,
        "ad_length": ad_len,
        "has_screenshots": has_ss,
    }


# ---------------------------------------------------------
# Piyasa fiyat aralığı tahmini (web yok → aralık + güven + gerekçe)
# ---------------------------------------------------------
def estimate_market_price_range(req: AnalyzeRequest, segment_code: str) -> Dict[str, Any]:
    """
    Amaç: ilan fiyatının 'piyasa altı/uygun/üstü/belirsiz' yorumuna zemin oluşturmak.
    NOT: Buradaki değerler "yaklaşık"tır → kesin fiyat iddiası değil; güven skoru döner.
    """
    v = req.vehicle
    listed = _parse_listed_price(req)

    # Segment baz median (2025 için) — KABA ANKRAJ (istersen env ile güncellersin)
    base_median_by_segment_2025 = {
        "B_HATCH": 750_000,
        "C_SEDAN": 1_150_000,
        "C_SUV": 1_550_000,
        "D_SEDAN": 1_850_000,
        "PREMIUM_D": 2_700_000,
        "E_SEGMENT": 4_000_000,
    }
    median = float(base_median_by_segment_2025.get(segment_code, 1_150_000))

    conf = 0.35
    reasons: List[str] = ["Segment bazlı kaba piyasa ankrajı kullanıldı (web verisi yok)."]

    # yıl/yaş etkisi (en büyük belirsizlik burada)
    if v.year:
        age = max(0, CURRENT_YEAR - v.year)
        # yeni → daha yüksek, yaşlandıkça düşer (TR enflasyon yüzünden bu çok oynar → conf düşük)
        age_factor = (0.94 ** min(age, 12))  # 6 yaş → ~0.69
        median *= age_factor
        conf += 0.18
        reasons.append(f"Yaş etkisi uygulandı: {age} yıl (fiyat belirsizliği yüksek).")
    else:
        reasons.append("Yıl bilgisi yok → piyasa tahmini güveni düşük.")

    # km etkisi
    km = int(v.mileage_km or 0)
    if km > 0:
        # referans: 60k km; üstüne çıktıkça düşür
        diff = km - 60_000
        steps = diff / 25_000.0
        km_factor = 1.0 - (0.03 * max(0.0, steps)) + (0.02 * max(0.0, -steps))
        km_factor = _clamp(km_factor, 0.55, 1.10)
        median *= km_factor
        conf += 0.12
        reasons.append("Km etkisi uygulandı (60k referans; yüksek km fiyatı düşürür).")
    else:
        reasons.append("Km bilgisi yok → piyasa tahmini güveni düşük.")

    # marka etkisi (likidite + talep)
    make_n = _norm(v.make)
    brand_mult = 1.0
    if any(x in make_n for x in ["toyota"]):
        brand_mult = 1.08
        reasons.append("Toyota talebi genelde güçlü → fiyat çarpanı hafif artırıldı.")
        conf += 0.05
    elif any(x in make_n for x in ["honda"]):
        brand_mult = 1.06
        reasons.append("Honda talebi genelde güçlü → fiyat çarpanı hafif artırıldı.")
        conf += 0.05
    elif any(x in make_n for x in ["vw", "volkswagen"]):
        brand_mult = 1.03
        reasons.append("VW talebi güçlü → fiyat çarpanı hafif artırıldı.")
        conf += 0.04
    elif any(x in make_n for x in ["fiat"]):
        brand_mult = 0.96
        reasons.append("Fiat/Egea tarafında fiyatlar daha rekabetçi olabiliyor → hafif düşürüldü.")
        conf += 0.03
    elif any(x in make_n for x in ["bmw", "mercedes", "audi"]):
        brand_mult = 1.05
        reasons.append("Premium algısı → fiyat çarpanı hafif artırıldı.")
        conf += 0.04

    median *= brand_mult

    # yakıt etkisi (LPG bazen fiyat kırar; hibrit/electric arttırabilir)
    fuel = _norm(v.fuel or "")
    if fuel == "lpg":
        median *= 0.97
        conf += 0.03
        reasons.append("LPG bazı alıcılarda fiyatı baskılayabilir → hafif düşürüldü.")
    elif fuel == "hybrid":
        median *= 1.03
        conf += 0.03
        reasons.append("Hybrid talebi yüksek olabiliyor → hafif artırıldı.")
    elif fuel == "electric":
        median *= 1.02
        conf += 0.02
        reasons.append("Elektriklide batarya/altyapı değişken → küçük ayar yapıldı.")
    elif fuel == "diesel":
        median *= 0.99
        conf += 0.02
        reasons.append("Dizelde km/DPF endişesi fiyatı etkileyebilir → küçük ayar yapıldı.")

    conf = float(_clamp(conf, 0.10, 0.75))

    # aralık genişliği: güven düştükçe daha geniş bant
    width = 0.22 + (0.40 * (1.0 - conf))  # conf 0.7 → ~0.34, conf 0.2 → ~0.54
    low = int(median * (1.0 - width))
    high = int(median * (1.0 + width))

    fairness = "belirsiz"
    price_tag = None
    if listed and listed > 50_000:
        if listed < low * 0.95:
            fairness = "piyasa_altı"
            price_tag = "Uygun"
        elif listed > high * 1.05:
            fairness = "piyasa_üstü"
            price_tag = "Yüksek"
        else:
            fairness = "piyasa_uygun"
            price_tag = "Normal"
        reasons.append("İlan fiyatı, tahmini piyasa bandı ile kıyaslandı.")
    else:
        reasons.append("İlan fiyatı yok → piyasa uygunluğu belirsiz.")
        price_tag = None

    return {
        "estimated_price_try_range": [low, high],
        "estimated_median_try": int(median),
        "confidence": conf,
        "fairness": fairness,      # piyasa_altı | piyasa_uygun | piyasa_üstü | belirsiz
        "price_tag": price_tag,    # Uygun | Normal | Yüksek | None
        "reasons": reasons,
        "listed_price_try": listed,
    }


# ---------------------------------------------------------
# Maliyet ve risk tahmini (uçuk olmayan)
# ---------------------------------------------------------
def estimate_costs(req: AnalyzeRequest) -> Dict[str, Any]:
    v = req.vehicle
    p = req.profile or Profile()

    segment_code = detect_segment(v.make, v.model)
    seg = SEGMENT_PROFILES.get(segment_code, SEGMENT_PROFILES["C_SEDAN"])

    listed_price = _parse_listed_price(req)

    age = None
    if v.year:
        age = max(0, CURRENT_YEAR - v.year)
    mileage = int(v.mileage_km or 0)

    base_min, base_max = seg["maintenance_yearly_range"]

    # yaş / km / kullanım çarpanları
    age_mult = 1.0
    if age is not None:
        if age >= 15:
            age_mult = 1.8
        elif age >= 10:
            age_mult = 1.4
        elif age >= 6:
            age_mult = 1.15

    km_mult = 1.0
    if mileage >= 250_000:
        km_mult = 1.8
    elif mileage >= 180_000:
        km_mult = 1.4
    elif mileage >= 120_000:
        km_mult = 1.2

    usage_mult = 1.0
    if p.usage == "city":
        usage_mult = 1.15
    elif p.usage == "highway":
        usage_mult = 0.95

    fuel = (v.fuel or p.fuel_preference or "").lower()
    fuel_mult_maint = 1.0
    fuel_risk = "orta"
    fuel_notes: List[str] = []

    if fuel == "diesel":
        fuel_mult_maint = 1.05
        fuel_notes.append("Dizelde yüksek km + şehir içi kullanım DPF/EGR/enjektör riskini artırır.")
        if p.usage == "city" and mileage >= 120_000:
            fuel_risk = "orta-yüksek (DPF/EGR/enjektör riski şehir içiyle artabilir)"
    elif fuel == "lpg":
        fuel_mult_maint = 0.98  # LPG yakıt ucuz ama bakım/ayar ihtiyacı olabilir → çok düşürme
        fuel_risk = "orta (montaj ve ayar kalitesi önemli, subap takibi şart)"
        fuel_notes.append("LPG’de montaj kalitesi + ayar + subap/kompresyon takibi kritik.")
    elif fuel in ("hybrid", "electric"):
        fuel_mult_maint = 0.9
        fuel_risk = "düşük-orta (batarya sağlığına bağlı)"
        fuel_notes.append("Hybrid/EV’de batarya sağlığı ve geçmişi kritik; doğru rapor istenmeli.")

    maint_min = int(base_min * age_mult * km_mult * usage_mult * fuel_mult_maint)
    maint_max = int(base_max * age_mult * km_mult * usage_mult * fuel_mult_maint)

    # Araç fiyatına göre bakım bandını kısıtla (uçuk değer engeli)
    if listed_price and listed_price > 100000:
        ratio_map = {
            "B_HATCH": (0.015, 0.06),
            "C_SEDAN": (0.015, 0.07),
            "C_SUV": (0.02, 0.08),
            "D_SEDAN": (0.02, 0.085),
            "PREMIUM_D": (0.025, 0.09),
            "E_SEGMENT": (0.03, 0.10),
        }
        r_min, r_max = ratio_map.get(segment_code, (0.015, 0.07))
        hard_min = int(listed_price * r_min)
        hard_max = int(listed_price * r_max)
        maint_min = max(maint_min, hard_min)
        maint_max = min(maint_max, hard_max)

        caps = {
            "B_HATCH": 45000,
            "C_SEDAN": 65000,
            "C_SUV": 85000,
            "D_SEDAN": 95000,
            "PREMIUM_D": 140000,
            "E_SEGMENT": 180000,
        }
        cap = caps.get(segment_code, 70000)
        maint_max = min(maint_max, cap)

        if maint_min > maint_max:
            maint_min = int(maint_max * 0.7)

    mid_maint = int((maint_min + maint_max) / 2) if maint_max else maint_min
    routine_est = int(mid_maint * 0.65)
    risk_reserve_est = max(0, mid_maint - routine_est)

    # Yakıt / enerji yıllık tahmini (TL bandı)
    km_year = max(0, int(p.yearly_km or 15000))
    km_factor = km_year / 15000.0
    km_factor = max(0.5, min(2.5, km_factor))

    fuel_base = {
        "B_HATCH": (18000, 32000),
        "C_SEDAN": (22000, 40000),
        "C_SUV": (26000, 48000),
        "D_SEDAN": (27000, 52000),
        "PREMIUM_D": (32000, 65000),
        "E_SEGMENT": (38000, 80000),
    }.get(segment_code, (22000, 42000))

    fuel_mult_cost = 1.0
    if fuel == "diesel":
        fuel_mult_cost = 0.9
    elif fuel == "lpg":
        fuel_mult_cost = 0.75
    elif fuel in ("hybrid", "electric"):
        fuel_mult_cost = 0.7

    fuel_min = int(fuel_base[0] * km_factor * fuel_mult_cost)
    fuel_max = int(fuel_base[1] * km_factor * fuel_mult_cost)

    if listed_price and listed_price > 100000:
        fuel_ratio_map = {
            "B_HATCH": (0.02, 0.12),
            "C_SEDAN": (0.025, 0.14),
            "C_SUV": (0.03, 0.16),
            "D_SEDAN": (0.03, 0.17),
            "PREMIUM_D": (0.035, 0.19),
            "E_SEGMENT": (0.04, 0.22),
        }
        fr_min, fr_max = fuel_ratio_map.get(segment_code, (0.025, 0.15))
        hard_fuel_min = int(listed_price * fr_min)
        hard_fuel_max = int(listed_price * fr_max)

        fuel_min = max(fuel_min, hard_fuel_min)
        fuel_max = min(fuel_max, hard_fuel_max)

        fuel_max = min(fuel_max, 220000)
        if fuel_min > fuel_max:
            fuel_min = int(fuel_max * 0.6)

    fuel_mid = int((fuel_min + fuel_max) / 2) if fuel_max else fuel_min

    # Trafik sigortası + kasko (çok kabaca) — şehir/hasarsızlık bilinmediği için band veriyoruz
    # Trafik sigortası: sabit banda yakın; kasko: araç değerine göre.
    traffic_min, traffic_max = (7000, 25000)
    kasko_min, kasko_max = (None, None)
    if listed_price and listed_price > 100000:
        # kasko oranı (segment + yaş/kasko)
        base_kasko_ratio = {
            "B_HATCH": (0.018, 0.040),
            "C_SEDAN": (0.020, 0.045),
            "C_SUV": (0.023, 0.052),
            "D_SEDAN": (0.025, 0.058),
            "PREMIUM_D": (0.028, 0.070),
            "E_SEGMENT": (0.032, 0.080),
        }.get(segment_code, (0.020, 0.055))
        kr_min, kr_max = base_kasko_ratio
        # yaş/kilometre risk ekle
        risk_add = 0.0
        if age and age >= 10:
            risk_add += 0.004
        if mileage >= 180_000:
            risk_add += 0.004
        kasko_min = int(listed_price * (kr_min + risk_add))
        kasko_max = int(listed_price * (kr_max + risk_add))
        kasko_min = max(kasko_min, 12000)
        kasko_max = min(kasko_max, 220000)
        if kasko_min > kasko_max:
            kasko_min = int(kasko_max * 0.75)

    # Risk seviyesi
    risk_level = "orta"
    risk_notes: List[str] = []

    if age is not None and age >= 15:
        risk_level = "yüksek"
        risk_notes.append("Araç yaşı yüksek; kronik masraf ihtimali artmış olabilir.")
    elif age is not None and age >= 10:
        risk_level = "orta-yüksek"
        risk_notes.append("Yaş nedeniyle büyük bakım gereksinimi çıkabilir.")

    if mileage >= 250_000:
        risk_level = "yüksek"
        risk_notes.append("Km çok yüksek; motor/şanzıman revizyon riski artmış.")
    elif mileage >= 180_000 and risk_level != "yüksek":
        risk_level = "orta-yüksek"
        risk_notes.append("Km yüksek; yürüyen aksam ve mekanik masraf ihtimali artmış.")

    if "yüksek" in fuel_risk:
        risk_level = "yüksek"

    if segment_code in ("PREMIUM_D", "E_SEGMENT") and ((age and age > 10) or mileage > 180_000):
        risk_notes.append("Premium sınıfta yaşlı/yüksek km araçların büyük masraf kalemleri daha pahalı olabilir.")

    return {
        "segment_code": segment_code,
        "segment_name": seg["name"],
        "age": age,
        "mileage_km": mileage,
        "listed_price_try": listed_price,
        "maintenance_yearly_try_min": maint_min,
        "maintenance_yearly_try_max": maint_max,
        "maintenance_routine_yearly_est": routine_est,
        "maintenance_risk_reserve_yearly_est": risk_reserve_est,
        "yearly_fuel_tr_min": fuel_min,
        "yearly_fuel_tr_max": fuel_max,
        "yearly_fuel_tr_mid": fuel_mid,
        "insurance_level": seg["insurance_level"],
        "traffic_insurance_band_tr": [traffic_min, traffic_max],
        "kasko_band_tr": [kasko_min, kasko_max] if (kasko_min and kasko_max) else None,
        "parts_availability": seg["parts"],
        "resale_speed": seg["resale"],
        "fuel_risk_comment": fuel_risk,
        "fuel_notes": fuel_notes,
        "risk_level": risk_level,
        "risk_notes": risk_notes,
        "segment_notes": seg.get("notes", []),
        "consumption_l_per_100km_band": {
            "B_HATCH": (5.5, 7.5),
            "C_SEDAN": (6.5, 9.0),
            "C_SUV": (7.5, 11.0),
            "D_SEDAN": (7.5, 11.0),
            "PREMIUM_D": (8.0, 12.5),
            "E_SEGMENT": (9.0, 15.0),
        }.get(segment_code, (6.5, 9.5)),
    }


def build_enriched_context(req: AnalyzeRequest) -> Dict[str, Any]:
    v = req.vehicle
    p = req.profile or Profile()

    seg_info = estimate_costs(req)
    anchors = find_anchor_matches(v.make, v.model, seg_info["segment_code"], limit=3)
    info_q = evaluate_info_quality(req)

    yearly_km_band = "orta"
    if p.yearly_km <= 8000:
        yearly_km_band = "düşük"
    elif p.yearly_km >= 30000:
        yearly_km_band = "yüksek"

    pricing = estimate_market_price_range(req, seg_info["segment_code"])

    # ilan metninden bazı bayraklar (kullanıcı yazdıysa)
    ad_text = (req.ad_description or "")
    flags = {
        "mentions_tramer": _bool_from_text(ad_text, ["tramer", "hasar kaydi", "hasar kaydı", "pert", "agir hasar", "ağır hasar"]),
        "mentions_boya_degisen": _bool_from_text(ad_text, ["boya", "degisen", "değişen", "lokal boya", "komple boya"]),
        "mentions_otomatik": _bool_from_text(ad_text, ["otomatik", "dsg", "cvt", "edc", "s tronic", "s-tronic"]),
        "mentions_lpg": _bool_from_text(ad_text, ["lpg", "tup", "tüp"]),
        "mentions_first_owner": _bool_from_text(ad_text, ["ilk sahibinden", "ilk sahibi", "tek kullanici", "tek kullanıcı"]),
        "mentions_service": _bool_from_text(ad_text, ["yetkili servis", "servis bakimli", "servis bakımlı", "bakim kaydi", "bakım kaydı", "fatura"]),
    }

    return {
        "segment": {
            "code": seg_info["segment_code"],
            "name": seg_info["segment_name"],
            "notes": seg_info.get("segment_notes", []),
        },
        "market": {
            "resale_speed": seg_info["resale_speed"],
            "parts_availability": seg_info["parts_availability"],
            "insurance_level": seg_info["insurance_level"],
        },
        "costs": {
            "listed_price_try": seg_info["listed_price_try"],
            "maintenance_yearly_try_min": seg_info["maintenance_yearly_try_min"],
            "maintenance_yearly_try_max": seg_info["maintenance_yearly_try_max"],
            "maintenance_routine_yearly_est": seg_info["maintenance_routine_yearly_est"],
            "maintenance_risk_reserve_yearly_est": seg_info["maintenance_risk_reserve_yearly_est"],
            "yearly_fuel_tr_min": seg_info["yearly_fuel_tr_min"],
            "yearly_fuel_tr_max": seg_info["yearly_fuel_tr_max"],
            "yearly_fuel_tr_mid": seg_info["yearly_fuel_tr_mid"],
            "traffic_insurance_band_tr": seg_info["traffic_insurance_band_tr"],
            "kasko_band_tr": seg_info["kasko_band_tr"],
            "consumption_l_per_100km_band": seg_info["consumption_l_per_100km_band"],
        },
        "pricing": pricing,
        "risk": {
            "fuel_risk_comment": seg_info["fuel_risk_comment"],
            "fuel_notes": seg_info.get("fuel_notes", []),
            "baseline_risk_level": seg_info["risk_level"],
            "risk_notes": seg_info["risk_notes"],
            "age": seg_info["age"],
            "mileage_km": seg_info["mileage_km"],
        },
        "profile": {
            "yearly_km": p.yearly_km,
            "yearly_km_band": yearly_km_band,
            "usage": p.usage,
            "fuel_preference": p.fuel_preference,
            "budget_try": getattr(p, "budget_try", None),
            "priorities": getattr(p, "priorities", None),
            "first_car": getattr(p, "first_car", None),
            "family": getattr(p, "family", None),
            "kids": getattr(p, "kids", None),
        },
        "info_quality": info_q,
        "flags": flags,
        "anchors_used": [
            {
                "key": a.get("key"),
                "segment": a.get("segment"),
                "chronic": a.get("chronic", []),
                "parts": a.get("parts"),
                "resale": a.get("resale"),
            }
            for a in anchors
        ],
    }


# ---------------------------------------------------------
# Kullanıcı prompt'u (LLM için)
# ---------------------------------------------------------
def build_user_content(req: AnalyzeRequest, mode: str) -> str:
    v = req.vehicle or Vehicle()
    p = req.profile or Profile()

    ad_text = (req.ad_description or "").strip()

    all_ss: List[str] = []
    if req.screenshot_base64:
        all_ss.append(req.screenshot_base64)
    if req.screenshots_base64:
        all_ss.extend([s for s in req.screenshots_base64 if s])

    enriched = build_enriched_context(req)

    ss_info = ""
    if all_ss:
        ss_info = (
            f"\nKullanıcı {len(all_ss)} adet ilan ekran görüntüsü ekledi. "
            "Bu görüntülerdeki fiyat, donanım, paket, boya-değişen ve hasar bilgileri varsa analizinde bunları da dikkate al. "
            "Görüntü içeriğini görmüyorsan, 'info_quality' ve 'assumptions' tarzı uyarı cümleleriyle belirsizliği açıkla."
        )

    if not (v.make.strip() or v.model.strip() or ad_text or all_ss):
        ad_text = "Kullanıcı çok az bilgi verdi. Türkiye ikinci el piyasasında genel kabul gören kriterlerle, varsayımsal ama faydalı bir değerlendirme yap."

    base_text = f"""
Kullanıcı Oto Analiz uygulamasında **{mode}** modunda analiz istiyor.

Araç bilgileri:
- Marka: {v.make or "-"}
- Model: {v.model or "-"}
- Paket/Trim: {v.trim or "-"}
- Motor: {v.engine or "-"}
- Şanzıman: {v.transmission or "-"}
- Gövde: {v.body_type or "-"}
- Yıl: {v.year or "-"}
- Kilometre: {v.mileage_km or "-"} km
- Yakıt: {v.fuel or p.fuel_preference}

Kullanım profili:
- Yıllık km: {p.yearly_km} km
- Kullanım tipi: {p.usage}
- Yakıt tercihi: {p.fuel_preference}
- Bütçe (varsa): {getattr(p, "budget_try", None) or "-"}
- Öncelikler (varsa): {getattr(p, "priorities", None) or "-"}

İlan açıklaması / kullanıcı notu:
{ad_text if ad_text else "-"}

--- Referans veri + segment emsal (BUNU ÖNEMLE KULLAN) ---
{json.dumps(enriched, ensure_ascii=False)}
---------------------------------------------------------
{ss_info}
""".strip()

    return base_text


# ---------------------------------------------------------
# JSON sağlamlaştırma yardımcıları
# ---------------------------------------------------------
def _ensure_list(d: Dict[str, Any], key: str, min_len: int, filler: str) -> None:
    v = d.get(key)
    if not isinstance(v, list):
        d[key] = []
        v = d[key]
    while len(v) < min_len:
        v.append(filler)


def _ensure_dict(d: Dict[str, Any], key: str) -> Dict[str, Any]:
    v = d.get(key)
    if not isinstance(v, dict):
        d[key] = {}
    return d[key]


def _coerce_int_in_range(x: Any, lo: int, hi: int, fallback: int) -> int:
    v = _safe_int(x)
    if v is None:
        return fallback
    return int(_clamp(float(v), float(lo), float(hi)))


def postprocess_premium_json(data: Dict[str, Any], req: AnalyzeRequest) -> Dict[str, Any]:
    """
    1) Band dışı masrafı düzelt
    2) preview.price_tag'i piyasa analizinden set et
    3) details başlıklarını garanti altına al
    """
    ctx = build_enriched_context(req)
    costs_ctx = ctx["costs"]
    pricing_ctx = ctx["pricing"]

    # cost_estimates zorunlu alanları banddan doldur
    ce = _ensure_dict(data, "cost_estimates")
    ce["yearly_maintenance_tr_min"] = int(costs_ctx["maintenance_yearly_try_min"])
    ce["yearly_maintenance_tr_max"] = int(costs_ctx["maintenance_yearly_try_max"])
    ce["yearly_fuel_tr_min"] = int(costs_ctx["yearly_fuel_tr_min"])
    ce["yearly_fuel_tr_max"] = int(costs_ctx["yearly_fuel_tr_max"])
    ce["insurance_level"] = ctx["market"]["insurance_level"]
    ce["insurance_band_tr"] = costs_ctx.get("traffic_insurance_band_tr")  # geriye dönük isim
    ce["traffic_insurance_band_tr"] = costs_ctx.get("traffic_insurance_band_tr")
    ce["kasko_band_tr"] = costs_ctx.get("kasko_band_tr")

    # yearly_maintenance_tr / yearly_fuel_tr band içinde olsun
    maint_min = int(costs_ctx["maintenance_yearly_try_min"])
    maint_max = int(costs_ctx["maintenance_yearly_try_max"])
    fuel_min = int(costs_ctx["yearly_fuel_tr_min"])
    fuel_max = int(costs_ctx["yearly_fuel_tr_max"])

    ce["yearly_maintenance_tr"] = _coerce_int_in_range(
        ce.get("yearly_maintenance_tr"),
        maint_min, maint_max,
        fallback=int((maint_min + maint_max) / 2),
    )
    ce["yearly_fuel_tr"] = _coerce_int_in_range(
        ce.get("yearly_fuel_tr"),
        fuel_min, fuel_max,
        fallback=int((fuel_min + fuel_max) / 2),
    )

    if not ce.get("notes"):
        ce["notes"] = "Bakım/yakıt/sigorta-kasko bandları segment + yaş + km + kullanım profiline göre tahmini aralık olarak verildi; gerçek değerler araç geçmişine göre değişebilir."

    # Preview -> price_tag (Uygun/Normal/Yüksek)
    preview = _ensure_dict(data, "preview")
    if preview.get("price_tag") is None:
        preview["price_tag"] = pricing_ctx.get("price_tag")

    # pricing alanını ekstra ekle (frontend görmezse sorun değil)
    data["pricing"] = pricing_ctx

    # risk_analysis garanti
    ra = _ensure_dict(data, "risk_analysis")
    if not isinstance(ra.get("chronic_issues"), list):
        ra["chronic_issues"] = []
    if not isinstance(ra.get("warnings"), list):
        ra["warnings"] = []

    # anchors chronic ekle
    chronic = [x for a in ctx["anchors_used"] for x in (a.get("chronic") or [])]
    for c in chronic:
        if c not in ra["chronic_issues"]:
            ra["chronic_issues"].append(c)
    ra["risk_level"] = ra.get("risk_level") or ctx["risk"]["baseline_risk_level"]

    # summary pros/cons minimum
    summary = _ensure_dict(data, "summary")
    if not isinstance(summary.get("pros"), list):
        summary["pros"] = []
    if not isinstance(summary.get("cons"), list):
        summary["cons"] = []
    _ensure_list(summary, "pros", 8, "İlan detayları netleştikçe daha doğru karar verilebilir.")
    _ensure_list(summary, "cons", 8, "Bakım geçmişi ve ekspertiz raporu görülmeden kesin kanaat vermek doğru olmaz.")

    # details başlıkları (her biri min 4)
    details = _ensure_dict(data, "details")
    for k in [
        "personal_fit",
        "maintenance_breakdown",
        "insurance_kasko",
        "parts_and_service",
        "resale_market",
        "negotiation_tips",
        "alternatives_same_segment",
    ]:
        if not isinstance(details.get(k), list):
            details[k] = []
        _ensure_list(details, k, 4, "Eksik veri nedeniyle genel rehber öneri verildi; detaylar ilan/ekspertiz ile netleşir.")

    # preview bullets minimum
    if not isinstance(preview.get("bullets"), list):
        preview["bullets"] = []
    _ensure_list(preview, "bullets", 4, "Ekspertiz + tramer + bakım kayıtları birlikte değerlendirilmelidir.")

    # result minimum uzunluk (satır hedefi)
    result = data.get("result", "")
    if not isinstance(result, str):
        result = str(result or "")
    if len(result.splitlines()) < 18:
        # kısa kaldıysa en azından ctx özetini ekle
        extra_lines = [
            "",
            "=== Hızlı Özet (Sistem) ===",
            f"- Segment: {ctx['segment']['name']}",
            f"- Piyasa uygunluğu (tahmini): {pricing_ctx.get('fairness')} | Güven: {pricing_ctx.get('confidence')}",
            f"- Yıllık bakım bandı: {maint_min:,} - {maint_max:,} TL".replace(",", "."),
            f"- Yıllık yakıt bandı: {fuel_min:,} - {fuel_max:,} TL".replace(",", "."),
            f"- Trafik sigortası bandı: {costs_ctx.get('traffic_insurance_band_tr')}",
            f"- Kasko bandı: {costs_ctx.get('kasko_band_tr')}",
            f"- İkinci el likidite: {ctx['market']['resale_speed']}",
            f"- Parça/usta erişimi: {ctx['market']['parts_availability']}",
        ]
        data["result"] = (result + "\n" + "\n".join(extra_lines)).strip()

    return data


def premium_quality_ok(data: Dict[str, Any]) -> bool:
    try:
        details = data.get("details", {})
        summary = data.get("summary", {})
        ra = data.get("risk_analysis", {})
        result = data.get("result", "")

        if not isinstance(result, str) or len(result.splitlines()) < 20:
            return False
        if not isinstance(summary.get("pros"), list) or len(summary["pros"]) < 8:
            return False
        if not isinstance(summary.get("cons"), list) or len(summary["cons"]) < 8:
            return False
        if not isinstance(ra.get("warnings"), list) or len(ra["warnings"]) < 6:
            return False

        for k in [
            "personal_fit",
            "maintenance_breakdown",
            "insurance_kasko",
            "parts_and_service",
            "resale_market",
            "negotiation_tips",
            "alternatives_same_segment",
        ]:
            if not isinstance(details.get(k), list) or len(details[k]) < 4:
                return False
        return True
    except:
        return False


# ---------------------------------------------------------
# Fallback JSON üreticileri (LLM patlarsa)
# ---------------------------------------------------------
def fallback_normal(req: AnalyzeRequest) -> Dict[str, Any]:
    v = req.vehicle
    title = f"{v.year or ''} {v.make} {v.model}".strip() or "Araç Analizi"
    return {
        "scores": {"overall_100": 70, "mechanical_100": 70, "body_100": 70, "economy_100": 70},
        "summary": {
            "short_comment": "Sınırlı bilgiye göre genel bir değerlendirme yapıldı.",
            "pros": [
                "Ekspertiz ve tramer ile detaylar netleştirildiğinde mantıklı bir tercih olabilir.",
                "Türkiye ikinci el piyasasındaki genel beklentilere göre nötr bir konumda görünüyor.",
            ],
            "cons": [
                "İlan detayları, bakım geçmişi ve km bilgisi tam bilinmeden kesin kanaat vermek doğru olmaz.",
                "Satın almadan önce mutlaka ekspertiz ve tramer sorgusu yapılmalı.",
            ],
            "estimated_risk_level": "orta",
        },
        "preview": {
            "title": title,
            "price_tag": None,
            "spoiler": "Genel değerlendirme hazır. Ekspertiz ve bakım geçmişi teyit edilmeden karar verilmemeli.",
            "bullets": [
                "Tramer/hasar kaydını kontrol et",
                "Bakım kayıtları ve km uyumuna bak",
                "Lastik, fren ve alt takım durumuna dikkat et",
            ],
        },
        "result": "Genel, nötr bir ikinci el değerlendirmesi sağlandı. Detaylı karar için ilan açıklaması, ekspertiz raporu ve tramer sorgusu mutlaka birlikte düşünülmelidir.",
    }


def fallback_premium(req: AnalyzeRequest) -> Dict[str, Any]:
    v = req.vehicle
    title = f"{v.year or ''} {v.make} {v.model}".strip() or "Premium Analiz"
    ctx = build_enriched_context(req)

    maint_min = int(ctx["costs"]["maintenance_yearly_try_min"])
    maint_max = int(ctx["costs"]["maintenance_yearly_try_max"])
    fuel_min = int(ctx["costs"]["yearly_fuel_tr_min"])
    fuel_max = int(ctx["costs"]["yearly_fuel_tr_max"])
    maint_mid = int((maint_min + maint_max) / 2)
    fuel_mid = int(ctx["costs"]["yearly_fuel_tr_mid"])

    chronic = [x for a in ctx["anchors_used"] for x in (a.get("chronic") or [])][:8]

    base = {
        "scores": {
            "overall_100": 75,
            "mechanical_100": 74,
            "body_100": 73,
            "economy_100": 70,
            "comfort_100": 72,
            "family_use_100": 76,
            "resale_100": 74,
        },
        "cost_estimates": {
            "yearly_maintenance_tr": maint_mid,
            "yearly_maintenance_tr_min": maint_min,
            "yearly_maintenance_tr_max": maint_max,
            "yearly_fuel_tr": fuel_mid,
            "yearly_fuel_tr_min": fuel_min,
            "yearly_fuel_tr_max": fuel_max,
            "insurance_level": ctx["market"]["insurance_level"],
            "insurance_band_tr": ctx["costs"]["traffic_insurance_band_tr"],
            "traffic_insurance_band_tr": ctx["costs"]["traffic_insurance_band_tr"],
            "kasko_band_tr": ctx["costs"]["kasko_band_tr"],
            "notes": "Bakım/yakıt/sigorta-kasko değerleri segment+yaş+km+kullanım profiline göre tahmini banddır.",
        },
        "risk_analysis": {
            "chronic_issues": chronic,
            "risk_level": ctx["risk"]["baseline_risk_level"],
            "warnings": ctx["risk"]["risk_notes"]
            + [ctx["risk"]["fuel_risk_comment"], "Satın almadan önce kapsamlı ekspertiz ve tramer sorgusu zorunludur."],
        },
        "summary": {
            "short_comment": "Segment emsali ve kullanım profilinize göre masraf ve risk bandı çıkarıldı; kesin karar için ekspertiz şart.",
            "pros": [
                f"Parça bulunabilirliği: {ctx['market']['parts_availability']}.",
                f"İkinci el likidite: {ctx['market']['resale_speed']}.",
                "Masraf bandı önceden öngörüldü (bakım + yakıt + sigorta/kasko).",
                "Piyasa uygunluğu tahmini yapıldı (güven skoru ile).",
                "Kronik riskler için kontrol adımları listelendi.",
                "Pazarlık için sorular ve kontrol listesi eklendi.",
                "Kullanıcı profiline uygunluk değerlendirmesi eklendi.",
                "Alternatif araç önerileri eklendi.",
            ],
            "cons": [
                "İlan/ekspertiz detayı yoksa bazı riskler belirsiz kalır.",
                "Tramer ve bakım geçmişi kötü çıkarsa masraf bandı yukarı kayabilir.",
                "Yüksek km/yaş kombinasyonu büyük bakım ihtimalini artırabilir.",
                "Sigorta/kasko primi şehir, yaş, hasarsızlık gibi değişkenlere çok duyarlıdır.",
                "LPG/dizel gibi sistemlerde bakım kalitesi kritik fark yaratır.",
                "Test sürüşü yapılmadan şanzıman/alt takım netleşmez.",
                "Kaporta işlem varsa değer kaybı etkisi olabilir.",
                "Satış hızı il/renk/donanım/tramer durumuna göre değişir.",
            ],
            "who_should_buy": "Bütçesini bilen, yıllık km’si ile segment masrafını kabul eden ve satın almadan önce ekspertiz+tramer yaptıracak kullanıcı için daha uygundur.",
        },
        "preview": {
            "title": title,
            "price_tag": ctx["pricing"].get("price_tag"),
            "spoiler": "Premium formatta masraf + piyasa + risk + uygunluk özeti hazır.",
            "bullets": [
                "Yıllık bakım/yakıt bandı çıkarıldı",
                "Trafik sigortası + kasko bandı tahmini verildi",
                "Piyasa uygunluğu (tahmini) ve pazarlık rehberi eklendi",
                "Kronik riskler + kontrol adımları listelendi",
            ],
        },
        "details": {
            "personal_fit": [
                "Kullanım tipi ve yıllık km’ye göre yakıt/maliyet dengesi yorumlandı.",
                "Bütçe hassasiyetin varsa, beklenmedik masraf payını göz önünde bulundur.",
                "Aile kullanımı varsa güvenlik + bagaj + arka sıra konforu öne çıkar.",
                "İlk araç ise sigorta/kasko primi bütçeyi zorlayabilir.",
            ],
            "maintenance_breakdown": [
                f"Yıllık bakım bandı: {maint_min:,}-{maint_max:,} TL (tahmini).".replace(",", "."),
                "Periyodik bakım + sarf (yağ/filtre/fren/lastik) düzenli takip edilmeli.",
                "Beklenmedik masraf için ayrıca pay ayır (özellikle yaş/km yükseldikçe).",
                "Test sürüşü + OBD tarama + alt takım kontrolü kritik.",
            ],
            "insurance_kasko": [
                f"Trafik sigortası bandı (tahmini): {ctx['costs']['traffic_insurance_band_tr']}.",
                f"Kasko bandı (tahmini): {ctx['costs']['kasko_band_tr']}.",
                "Primler şehir, sürücü yaşı, hasarsızlık ve araç geçmişine göre ciddi değişir.",
                "Teklif alıp karşılaştırmadan karar verme; teminatları aynı tut.",
            ],
            "parts_and_service": [
                f"Parça/usta erişimi: {ctx['market']['parts_availability']}.",
                "Düzenli bakım kayıtları masraf sürprizini azaltır.",
                "Premium araçlarda orijinal parça maliyeti belirgin yüksektir; doğru servis seçimi kritiktir.",
                "İyi özel servis + doğru parça seçimi bütçeyi dengeler.",
            ],
            "resale_market": [
                f"İkinci elde satış hızı (genel): {ctx['market']['resale_speed']}.",
                "Bakım kayıtları + düşük tramer satarken en büyük artı.",
                "Renk/donanım/paket ve il bazlı talep satış süresini etkiler.",
                "Kaporta işlem/hasar satış hızını düşürebilir.",
            ],
            "negotiation_tips": [
                "Ekspertizde çıkan kalemleri (lastik, fren, alt takım, büyük bakım) pazarlığa yaz.",
                "Tramer/hasar kaydını görmeden fiyat konuşma.",
                "Bakım faturasını ve tarihini iste; sözle yetinme.",
                "Piyasa bandı ile kıyaslayıp teklif ver.",
            ],
            "alternatives_same_segment": [
                "Aynı segmentte 2-3 emsali mutlaka kıyasla (km/yıl/hasar).",
                "Daha düşük km’li emsal, toplam maliyette daha avantajlı olabilir.",
                "Yakıt tercihini yıllık km’ye göre seç (düşük km’de dizel gereksiz kalabilir).",
                "Premium beklentin yoksa, aynı bütçeyle daha yeni/az km alternatif bulunabilir.",
            ],
        },
        "result": "",
    }

    base["result"] = (
        "=== Premium Rehber Analiz ===\n"
        f"- Bilgi seviyesi: {ctx['info_quality']['level']} (eksikler: {ctx['info_quality']['missing_fields']})\n"
        f"- Segment: {ctx['segment']['name']} | Likidite: {ctx['market']['resale_speed']} | Parça: {ctx['market']['parts_availability']}\n"
        f"- Piyasa uygunluğu (tahmini): {ctx['pricing']['fairness']} | Güven: {ctx['pricing']['confidence']}\n\n"
        "1) Piyasa & Fiyat\n"
        f"- Tahmini piyasa bandı: {ctx['pricing']['estimated_price_try_range']}\n"
        "- Fiyat yorumu, web/ilan taraması olmadan yalnızca segment+yaş+km ankrajına dayanır.\n\n"
        "2) Yıllık Masraf (TRY)\n"
        f"- Bakım bandı: {maint_min:,}-{maint_max:,} TL\n".replace(",", ".") +
        f"- Yakıt bandı: {fuel_min:,}-{fuel_max:,} TL\n".replace(",", ".") +
        f"- Trafik sigortası: {ctx['costs']['traffic_insurance_band_tr']}\n"
        f"- Kasko: {ctx['costs']['kasko_band_tr']}\n\n"
        "3) Risk & Kronikler\n"
        f"- Risk seviyesi: {ctx['risk']['baseline_risk_level']} | Notlar: {ctx['risk']['risk_notes']}\n"
        f"- Yakıt sistemi riski: {ctx['risk']['fuel_risk_comment']}\n"
        "- Satın almadan önce: OBD tarama + kompresyon/yağ kaçak kontrolü + alt takım kontrolü önerilir.\n\n"
        "4) Pazarlık\n"
        "- Ekspertiz raporundaki masrafları kalem kalem pazarlığa yaz.\n"
        "- Tramer ve bakım faturası olmadan 'temiz' beyanı yeterli değildir.\n\n"
        "5) Sonuç\n"
        "- Ekspertiz + tramer + bakım geçmişi iyi çıkarsa; masraf bandı yönetilebilir bir aralıkta kalır.\n"
        "- Belirsizlikler netleşmeden kesin karar verme.\n"
    )

    return postprocess_premium_json(base, req)


def fallback_manual(req: AnalyzeRequest) -> Dict[str, Any]:
    return fallback_normal(req)


def fallback_compare(req: CompareRequest) -> Dict[str, Any]:
    left_title = (f"{req.left.vehicle.make} {req.left.vehicle.model}").strip() or "Sol araç"
    right_title = (f"{req.right.vehicle.make} {req.right.vehicle.model}").strip() or "Sağ araç"
    return {
        "better_overall": "left",
        "summary": f"{left_title}, varsayılan olarak biraz daha dengeli kabul edildi. Ancak iki araç için de ekspertiz ve tramer şart.",
        "left_pros": [
            f"{left_title} genel kullanım ve masraf dengesi açısından daha öngörülebilir olabilir.",
            "Aile ve karışık kullanım için uygun bir profil sunabilir.",
        ],
        "left_cons": ["Gerçek durum eksper raporu ve bakım geçmişine göre netleşecektir."],
        "right_pros": [f"{right_title} doğru bakımla mantıklı bir alternatif olabilir."],
        "right_cons": ["Masraf ve risk tarafında daha dikkatli inceleme gerektirebilir."],
        "use_cases": {
            "family_use": f"Aile kullanımı için {left_title} biraz daha avantajlı varsayılmıştır.",
            "long_distance": "Her iki araç da düzenli bakım ile uzun yolda kullanılabilir.",
            "city_use": "Şehir içi kullanımda şanzıman tipi ve yakıt tüketimi önemli belirleyicilerdir.",
        },
    }


def fallback_otobot(question: str) -> Dict[str, Any]:
    return {
        "answer": "Bütçe, yıllık km, kullanım tipi ve beklentilere göre segment seçmek en mantıklısıdır. Yıllık km yüksekse ekonomik dizel/LPG; düşükse benzinli/hibrit seçenekler daha mantıklı olabilir. Satın almadan önce ekspertiz ve tramer sorgusunu mutlaka yaptır.",
        "suggested_segments": ["C-sedan", "C-SUV", "B-SUV"],
        "example_models": ["Toyota Corolla", "Renault Megane", "Honda Civic", "Hyundai Tucson"],
    }


# ---------------------------------------------------------
# LLM JSON çağrısı (hata olursa fallback) + Premium Detay Guard
# ---------------------------------------------------------
def call_llm_json(
    model_name: str,
    system_prompt: str,
    user_content: str,
    mode: str,
    req: Any,
) -> Dict[str, Any]:
    if client is None:
        if mode == "normal":
            return fallback_normal(req)
        if mode == "premium":
            return fallback_premium(req)
        if mode == "manual":
            return fallback_manual(req)
        if mode == "compare":
            return fallback_compare(req)
        if mode == "otobot":
            return fallback_otobot(user_content)
        return fallback_normal(req)

    try:
        max_tokens = MAX_TOKENS_NORMAL
        temperature = TEMP_NORMAL
        if mode == "premium":
            max_tokens = MAX_TOKENS_PREMIUM
            temperature = TEMP_PREMIUM
        elif mode == "compare":
            max_tokens = MAX_TOKENS_COMPARE
            temperature = TEMP_COMPARE
        elif mode == "otobot":
            max_tokens = MAX_TOKENS_OTOBOT
            temperature = TEMP_OTOBOT

        resp = client.chat.completions.create(
            model=model_name,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        content = resp.choices[0].message.content
        if isinstance(content, str):
            data = json.loads(content)
        else:
            data = content  # type: ignore[assignment]

        # Premium postprocess + kalite kontrol
        if mode == "premium" and isinstance(data, dict):
            data = postprocess_premium_json(data, req)

            if not premium_quality_ok(data):
                # Detay guard: eksikleri tamamlat
                fix_prompt = SYSTEM_PROMPT_PREMIUM + """
EK GÖREV (DETAY GUARD):
- Gönderdiğim JSON'u bozmadan geliştir: alanları genişlet, maddeleri artır.
- Özellikle: warnings>=8, pros>=10, cons>=10, her details listesi >=6 madde,
  result >= 30 satır (başlık + madde formatında).
- Masraf bandlarının dışına çıkma. Piyasa bandı için sadece pricing alanını referans al.
- Sadece GEÇERLİ JSON döndür.
"""
                fix_user = f"""
Aşağıdaki JSON premium analiz çıktısı kısa/eksik kalmış olabilir.
Eksikleri tamamla ve daha uzun, daha bilgili hale getir.

ORİJİNAL JSON:
{json.dumps(data, ensure_ascii=False)}
"""
                resp2 = client.chat.completions.create(
                    model=model_name,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": fix_prompt},
                        {"role": "user", "content": fix_user},
                    ],
                    max_tokens=MAX_TOKENS_PREMIUM,
                    temperature=TEMP_PREMIUM,
                )
                content2 = resp2.choices[0].message.content
                if isinstance(content2, str):
                    data2 = json.loads(content2)
                else:
                    data2 = content2  # type: ignore[assignment]
                if isinstance(data2, dict):
                    data = postprocess_premium_json(data2, req)

        return data  # type: ignore[return-value]

    except Exception as e:
        print(f"LLM hatası ({mode}): {e}")
        if mode == "normal":
            return fallback_normal(req)
        if mode == "premium":
            return fallback_premium(req)
        if mode == "manual":
            return fallback_manual(req)
        if mode == "compare":
            return fallback_compare(req)
        if mode == "otobot":
            return fallback_otobot(user_content)
        return fallback_normal(req)


# ---------------------------------------------------------
# System promptlar
# ---------------------------------------------------------
SYSTEM_PROMPT_NORMAL = """
Sen 'Oto Analiz' uygulaması için çalışan bir araç ilanı analiz asistanısın.

ÇIKTIYI SADECE GEÇERLİ BİR JSON OLARAK DÖN. ŞABLON:

{
  "scores": {
    "overall_100": 0,
    "mechanical_100": 0,
    "body_100": 0,
    "economy_100": 0
  },
  "summary": {
    "short_comment": "",
    "pros": [],
    "cons": [],
    "estimated_risk_level": "orta"
  },
  "preview": {
    "title": "",
    "price_tag": null,
    "spoiler": "",
    "bullets": []
  },
  "result": ""
}

Kurallar:
- Tüm alanlar mutlaka dolu olsun (boşsa bile mantıklı bir metin üret).
- PREVIEW (Keşfet):
  - 'alınır', 'alınmaz', 'sakın', 'tehlikeli' gibi kelimeleri kullanma.
  - Fiyat için rakam yazma; sadece 'Uygun', 'Normal', 'Yüksek' gibi etiket veya null kullan.
- Son kullanıcıya gösterilecek ana açıklama "result" alanında olsun.
- Dil: Türkçe.
"""

SYSTEM_PROMPT_PREMIUM = """
Sen 'Oto Analiz' uygulamasının PREMIUM analiz asistanısın.
Hedef: kısa genel cümle değil; kullanıcıya gerçekten rehberlik eden UZUN, BİLGİLİ, MADDELİ premium rapor.

Sana sağlanan JSON içinde özellikle şu alanları kullan:
- segment: code, name, notes
- market: resale_speed, parts_availability, insurance_level
- costs: listed_price_try, maintenance_yearly_try_min/max, maintenance_routine_yearly_est, maintenance_risk_reserve_yearly_est,
         yearly_fuel_tr_min/max/mid, traffic_insurance_band_tr, kasko_band_tr, consumption_l_per_100km_band
- pricing: estimated_price_try_range, confidence, fairness, price_tag, reasons
- risk: baseline_risk_level, risk_notes, fuel_risk_comment, fuel_notes, age, mileage_km
- profile: yearly_km, yearly_km_band, usage, fuel_preference, budget_try, priorities, first_car, family, kids
- info_quality: level, missing_fields, has_screenshots
- anchors_used: chronic listesi vb.
- flags: ilan metninden çıkan ipuçları

Bilgi seviyesi:
- info_quality.level 'düşük' ise: result başında NET uyarı ver, tahminleri daha geniş aralık + daha temkinli anlat.
- 'orta' veya 'yüksek' ise daha detaylı ve yönlendirici ol.

Masraf tahminleri:
- Bakım maliyeti için mutlaka maintenance_yearly_try_min/max bandını KORU. Dışına çıkma.
- Yakıt için yearly_fuel_tr_min/max bandını KORU. Dışına çıkma.
- Trafik sigortası bandını traffic_insurance_band_tr ile özetle.
- Kasko bandı varsa kasko_band_tr ile özetle.
- Uçuk rakam yazma; bandlar dışına kesinlikle çıkma.

Piyasa fiyatı:
- pricing.estimated_price_try_range bandını kullanarak ilan fiyatını yorumla.
- pricing.confidence düşükse bunu açıkça yaz (web/emsal taraması olmadan sınırlı güven).

Kişiye uygunluk:
- profile.yearly_km_band, usage, fuel_preference ve (varsa) priorities üzerinden "kime uygun/kime uygun değil" üret.
- Düşük yıllık km'de dizel gereksiz olabilir; şehir içinde DPF/otomatik/ tüketim riskini anlat.
- Aile/çocuk varsa güvenlik, arka sıra, bagaj, konforu öne çıkar.

Kronik & risk:
- anchors_used chronic maddelerini risk_analysis.chronic_issues içine ekle.
- risk_analysis.warnings en az 8 madde olsun ve “nasıl kontrol edilir?” cümlesi içersin.

JSON ŞABLONU:

{
  "scores": {
    "overall_100": 0,
    "mechanical_100": 0,
    "body_100": 0,
    "economy_100": 0,
    "comfort_100": 0,
    "family_use_100": 0,
    "resale_100": 0
  },
  "cost_estimates": {
    "yearly_maintenance_tr": 0,
    "yearly_maintenance_tr_min": 0,
    "yearly_maintenance_tr_max": 0,
    "yearly_fuel_tr": 0,
    "yearly_fuel_tr_min": 0,
    "yearly_fuel_tr_max": 0,
    "insurance_level": "orta",
    "insurance_band_tr": null,
    "traffic_insurance_band_tr": null,
    "kasko_band_tr": null,
    "notes": ""
  },
  "risk_analysis": {
    "chronic_issues": [],
    "risk_level": "orta",
    "warnings": []
  },
  "summary": {
    "short_comment": "",
    "pros": [],
    "cons": [],
    "who_should_buy": ""
  },
  "preview": {
    "title": "",
    "price_tag": null,
    "spoiler": "",
    "bullets": []
  },
  "details": {
    "personal_fit": [],
    "maintenance_breakdown": [],
    "insurance_kasko": [],
    "parts_and_service": [],
    "resale_market": [],
    "negotiation_tips": [],
    "alternatives_same_segment": []
  },
  "result": ""
}

Kurallar (Premium kalite şartı):
- summary.pros >= 10, summary.cons >= 10
- risk_analysis.warnings >= 8 (her maddede “nasıl kontrol edilir” bilgisi olsun)
- details içindeki HER liste >= 6 madde
- result: en az 30 satır, başlık + madde karışımı, kullanıcıya adım adım rehber olacak şekilde.
- PREVIEW (Keşfet):
  - 'alınır', 'alınmaz', 'sakın', 'tehlikeli' gibi kelimeleri kullanma.
  - Fiyat için rakam yazma; sadece 'Uygun', 'Normal' veya 'Yüksek' gibi etiket ya da null kullan.
- Dil: Türkçe, sadece JSON dön.
"""

SYSTEM_PROMPT_MANUAL = """
Sen 'Oto Analiz' uygulamasında KULLANICININ KENDİ ARACI için manuel analiz yapan asistansın.
Çıktı formatın NORMAL analizle aynıdır.

Kurallar:
- PREVIEW nötr olsun, 'alınır/alınmaz' yazma.
- Bilgi azsa bile; ekspertiz, tramer ve bakım geçmişi gibi genel tavsiyeler ver.
- Dil: Türkçe, sadece JSON.
"""

SYSTEM_PROMPT_COMPARE = """
Sen 'Oto Analiz' uygulaması için ARAÇ KARŞILAŞTIRMA asistanısın.

Sadece JSON döndür:

{
  "better_overall": "left" | "right",
  "summary": "",
  "left_pros": [],
  "left_cons": [],
  "right_pros": [],
  "right_cons": [],
  "use_cases": {
    "family_use": "",
    "long_distance": "",
    "city_use": ""
  }
}

Dil: Türkçe, sadece JSON.
"""

SYSTEM_PROMPT_OTOBOT = """
Sen 'Oto Analiz' uygulamasının OTOBOT isimli araç alma rehberisin.

Çıktı sadece JSON:
{
  "answer": "",
  "suggested_segments": [],
  "example_models": []
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
    user_content = build_user_content(req, mode="normal")
    return call_llm_json(
        model_name=OPENAI_MODEL_NORMAL,
        system_prompt=SYSTEM_PROMPT_NORMAL,
        user_content=user_content,
        mode="normal",
        req=req,
    )


# ---------------------------------------------------------
# PREMIUM ANALİZ
# ---------------------------------------------------------
@app.post("/premium_analyze")
async def premium_analyze(req: AnalyzeRequest) -> Dict[str, Any]:
    user_content = build_user_content(req, mode="premium")
    data = call_llm_json(
        model_name=OPENAI_MODEL_PREMIUM,
        system_prompt=SYSTEM_PROMPT_PREMIUM,
        user_content=user_content,
        mode="premium",
        req=req,
    )
    # Son güvenlik: postprocess (LLM zaten yaptıysa idempotent)
    if isinstance(data, dict):
        data = postprocess_premium_json(data, req)
    return data


# ---------------------------------------------------------
# MANUEL / KENDİ ARACIN ANALİZİ
# ---------------------------------------------------------
@app.post("/manual_analyze")
@app.post("/manual")
async def manual_analyze(req: AnalyzeRequest) -> Dict[str, Any]:
    user_content = build_user_content(req, mode="manual")
    return call_llm_json(
        model_name=OPENAI_MODEL_NORMAL,
        system_prompt=SYSTEM_PROMPT_MANUAL,
        user_content=user_content,
        mode="manual",
        req=req,
    )


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

    return call_llm_json(
        model_name=OPENAI_MODEL_COMPARE,
        system_prompt=SYSTEM_PROMPT_COMPARE,
        user_content=user_content,
        mode="compare",
        req=req,
    )


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

    user_content = question

    return call_llm_json(
        model_name=OPENAI_MODEL_OTOBOT,
        system_prompt=SYSTEM_PROMPT_OTOBOT,
        user_content=user_content,
        mode="otobot",
        req=req,
    )
