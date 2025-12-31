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


class Vehicle(BaseModel):
    make: str = ""
    model: str = ""
    year: Optional[int] = Field(None, ge=1980, le=2035)
    mileage_km: Optional[int] = Field(None, ge=0)
    fuel: Optional[str] = None       # gasoline / diesel / lpg / hybrid / electric


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


def _parse_listed_price(req: AnalyzeRequest) -> Optional[int]:
    # Flutter context: listed_price_text
    try:
        txt = (req.context or {}).get("listed_price_text") or ""
        v = _digits(str(txt))
        if v and v > 10000:
            return v
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
# ANCHOR MODEL VERİTABANI (emsal)
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

    if len(missing) <= 1 and ad_len >= 120:
        level = "yüksek"
    elif len(missing) <= 3 and ad_len >= 40:
        level = "orta"
    else:
        level = "düşük"

    return {
        "level": level,
        "missing_fields": missing,
        "ad_length": ad_len,
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
        age = max(0, 2025 - v.year)
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

    if fuel == "diesel":
        fuel_mult_maint = 1.05
        if p.usage == "city" and mileage >= 120_000:
            fuel_risk = "orta-yüksek (DPF/EGR/enjektör riski şehir içiyle artabilir)"
    elif fuel == "lpg":
        fuel_mult_maint = 0.95
        fuel_risk = "orta (montaj ve ayar kalitesi önemli, subap takibi şart)"
    elif fuel in ("hybrid", "electric"):
        fuel_mult_maint = 0.9
        fuel_risk = "düşük-orta (batarya sağlığına bağlı)"

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
            "C_SEDAN": 55000,
            "C_SUV": 75000,
            "D_SEDAN": 85000,
            "PREMIUM_D": 130000,
            "E_SEGMENT": 160000,
        }
        cap = caps.get(segment_code, 60000)
        maint_max = min(maint_max, cap)

        if maint_min > maint_max:
            maint_min = int(maint_max * 0.7)

    mid_maint = int((maint_min + maint_max) / 2) if maint_max else maint_min
    routine_est = int(mid_maint * 0.65)
    risk_reserve_est = mid_maint - routine_est

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

        fuel_max = min(fuel_max, 180000)
        if fuel_min > fuel_max:
            fuel_min = int(fuel_max * 0.6)

    fuel_mid = int((fuel_min + fuel_max) / 2) if fuel_max else fuel_min

    # Sigorta bandı (çok kabaca)
    ins_min = None
    ins_max = None
    if listed_price and listed_price > 100000:
        ins_ratio_map = {
            "B_HATCH": (0.015, 0.04),
            "C_SEDAN": (0.017, 0.045),
            "C_SUV": (0.02, 0.05),
            "D_SEDAN": (0.022, 0.055),
            "PREMIUM_D": (0.025, 0.065),
            "E_SEGMENT": (0.03, 0.075),
        }
        ir_min, ir_max = ins_ratio_map.get(segment_code, (0.02, 0.05))
        ins_min = int(listed_price * ir_min)
        ins_max = int(listed_price * ir_max)
        ins_min = max(ins_min, 5000)
        ins_max = min(ins_max, 160000)
        if ins_min > ins_max:
            ins_min = int(ins_max * 0.7)

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

    if segment_code in ("PREMIUM_D", "E_SEGMENT") and (age and age > 10 or mileage > 180_000):
        risk_notes.append("Premium sınıfta yaşlı/yüksek km araçların büyük masraf kalemleri çok pahalı olabilir.")

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
        "insurance_band_tr": [ins_min, ins_max] if (ins_min and ins_max) else None,
        "parts_availability": seg["parts"],
        "resale_speed": seg["resale"],
        "fuel_risk_comment": fuel_risk,
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
            "insurance_band_tr": seg_info["insurance_band_tr"],
            "consumption_l_per_100km_band": seg_info["consumption_l_per_100km_band"],
        },
        "risk": {
            "fuel_risk_comment": seg_info["fuel_risk_comment"],
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
        },
        "info_quality": info_q,
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
            "Bu görüntülerdeki fiyat, donanım, paket, boya-değişen ve hasar bilgileri varsa analizinde bunları da dikkate al."
        )

    if not (v.make.strip() or v.model.strip() or ad_text or all_ss):
        ad_text = "Kullanıcı çok az bilgi verdi. Türkiye ikinci el piyasasında genel kabul gören kriterlerle, varsayımsal ama faydalı bir değerlendirme yap."

    base_text = f"""
Kullanıcı Oto Analiz uygulamasında **{mode}** modunda analiz istiyor.

Araç bilgileri:
- Marka: {v.make or "-"}
- Model: {v.model or "-"}
- Yıl: {v.year or "-"}
- Kilometre: {v.mileage_km or "-"} km
- Yakıt: {v.fuel or p.fuel_preference}

Kullanım profili:
- Yıllık km: {p.yearly_km} km
- Kullanım tipi: {p.usage}
- Yakıt tercihi: {p.fuel_preference}

İlan açıklaması / kullanıcı notu:
{ad_text if ad_text else "-"}

--- Referans veri + segment emsal (BUNU ÖNEMLE KULLAN) ---
{json.dumps(enriched, ensure_ascii=False)}
---------------------------------------------------------
{ss_info}
""".strip()

    return base_text


# ---------------------------------------------------------
# Fallback JSON üreticileri (LLM patlarsa)
# ---------------------------------------------------------
def fallback_normal(req: AnalyzeRequest) -> Dict[str, Any]:
    v = req.vehicle
    title = f"{v.year or ''} {v.make} {v.model}".strip() or "Araç Analizi"

    return {
        "scores": {
            "overall_100": 70,
            "mechanical_100": 70,
            "body_100": 70,
            "economy_100": 70
        },
        "summary": {
            "short_comment": "Sınırlı bilgiye göre genel bir değerlendirme yapıldı.",
            "pros": [
                "Ekspertiz ve tramer ile detaylar netleştirildiğinde mantıklı bir tercih olabilir.",
                "Türkiye ikinci el piyasasındaki genel beklentilere göre nötr bir konumda görünüyor."
            ],
            "cons": [
                "İlan detayları, bakım geçmişi ve km bilgisi tam bilinmeden kesin kanaat vermek doğru olmaz.",
                "Satın almadan önce mutlaka ekspertiz ve tramer sorgusu yapılmalı."
            ],
            "estimated_risk_level": "orta"
        },
        "preview": {
            "title": title,
            "price_tag": None,
            "spoiler": "Genel değerlendirme hazır. Ekspertiz ve bakım geçmişi teyit edilmeden karar verilmemeli.",
            "bullets": [
                "Tramer/hasar kaydını kontrol et",
                "Bakım kayıtları ve km uyumuna bak",
                "Lastik, fren ve alt takım durumuna dikkat et"
            ]
        },
        "result": "Genel, nötr bir ikinci el değerlendirmesi sağlandı. Detaylı karar için ilan açıklaması, ekspertiz raporu ve tramer sorgusu mutlaka birlikte düşünülmelidir."
    }


def fallback_premium(req: AnalyzeRequest) -> Dict[str, Any]:
    v = req.vehicle
    title = f"{v.year or ''} {v.make} {v.model}".strip() or "Premium Analiz"
    ctx = build_enriched_context(req)

    maint_mid = int(
        (ctx["costs"]["maintenance_yearly_try_min"] + ctx["costs"]["maintenance_yearly_try_max"]) / 2
    )
    fuel_mid = ctx["costs"]["yearly_fuel_tr_mid"]

    chronic = [x for a in ctx["anchors_used"] for x in (a.get("chronic") or [])][:6]

    return {
        "scores": {
            "overall_100": 75,
            "mechanical_100": 74,
            "body_100": 73,
            "economy_100": 70,
            "comfort_100": 72,
            "family_use_100": 76,
            "resale_100": 74
        },
        "cost_estimates": {
            "yearly_maintenance_tr": maint_mid,
            "yearly_maintenance_tr_min": ctx["costs"]["maintenance_yearly_try_min"],
            "yearly_maintenance_tr_max": ctx["costs"]["maintenance_yearly_try_max"],
            "yearly_fuel_tr": fuel_mid,
            "yearly_fuel_tr_min": ctx["costs"]["yearly_fuel_tr_min"],
            "yearly_fuel_tr_max": ctx["costs"]["yearly_fuel_tr_max"],
            "insurance_level": ctx["market"]["insurance_level"],
            "insurance_band_tr": ctx["costs"]["insurance_band_tr"],
            "notes": "Bakım ve yakıt maliyeti segment + yaş + km + kullanım profiline göre tahmini band olarak verilmiştir."
        },
        "risk_analysis": {
            "chronic_issues": chronic,
            "risk_level": ctx["risk"]["baseline_risk_level"],
            "warnings": ctx["risk"]["risk_notes"] + [
                ctx["risk"]["fuel_risk_comment"],
                "Satın almadan önce kapsamlı ekspertiz ve tramer sorgusu zorunlu kabul edilmelidir."
            ]
        },
        "summary": {
            "short_comment": "Segment emsali ve kullanım profilinize göre dengeli, fakat klasik ikinci el riskler barındıran bir araç görünümünde.",
            "pros": [
                f"Parça bulunabilirliği: {ctx['market']['parts_availability']}.",
                f"İkinci el piyasasında likidite: {ctx['market']['resale_speed']}.",
                "Kullanım profilinize göre bakım/yakıt bandı önceden tahmin edildi."
            ],
            "cons": [
                "Gerçek masraflar mevcut bakım geçmişine ve önceki kullanıcıya göre değişebilir.",
                "Ekspertiz ve tramer sonuçları görülmeden net karar verilmemeli."
            ],
            "who_should_buy": "Bütçesini bilen, yıllık km’si ile segmentin masraf seviyesini kabul edebilen ve satın almadan önce detaylı ekspertiz yaptırmayı planlayan kullanıcılar için daha uygundur."
        },
        "preview": {
            "title": title,
            "price_tag": None,
            "spoiler": "Premium formatta, segment emsallerine dayalı detaylı masraf ve risk özeti.",
            "bullets": [
                "Bakım + yakıt için yıllık tahmini band verildi",
                "Sigorta seviyesi ve olası prim aralığı özetlendi",
                "Yedek parça ve ikinci el piyasası hakkında yorum yapıldı"
            ]
        },
        "details": {
            "personal_fit": [
                "Yıllık km ve kullanım tipinize göre, araç şehir içi / uzun yol dengesine uygunluğu açısından değerlendirildi.",
                "Eğer öğrencisin ya da ilk aracın ise, sigorta ve kasko primleri bütçe üzerinde baskı yaratabilir.",
                "Aile kullanımı planlıyorsan, bagaj hacmi, arka sıra konforu ve güvenlik donanımları öne çıkar."
            ],
            "maintenance_breakdown": [
                f"Periyodik bakım için ayrılması önerilen ortalama pay: {int(maint_mid*0.6)} TL civarı.",
                f"Beklenmedik masraflar için ayrılması önerilen yıllık risk payı: {int(maint_mid*0.4)} TL civarı.",
                "Yüksek km veya yaş varsa, büyük bakım (triger, debriyaj, yürüyen aksam) için ekstra bütçe ayırmak gerekir."
            ],
            "insurance_kasko": [
                f"Sigorta seviyesi: {ctx['market']['insurance_level']} segmentinde.",
                "Genç sürücü / ilk sigorta ise primler ortalamanın üstünde olabilir.",
                "Hasarsızlık indirimi ve bulunulan şehir, primleri ciddi şekilde etkiler."
            ],
            "parts_and_service": [
                f"Yedek parça bulunabilirliği: {ctx['market']['parts_availability']} seviyesinde.",
                "Yetkili servis yerine, işini bilen özel servislerle masraflar daha makul tutulabilir.",
                "Premium markalarda orijinal parça maliyetleri belirgin şekilde yüksektir; yan sanayi/çıkma tercihleri iyi tartılmalıdır."
            ],
            "resale_market": [
                f"İkinci el satılabilirlik: {ctx['market']['resale_speed']} hızda değerlendirilebilir.",
                "Düzenli bakım kayıtları ve temiz tramer, satarken büyük avantaj sağlar.",
                "Segmentin talep gördüğü şehirlerde, satış süresi daha kısa olabilir."
            ],
            "negotiation_tips": [
                "Ekspertiz raporundaki kusurları (boya, değişen, mekanik masraf) mutlaka fiyat pazarlığına yansıt.",
                "Tramer tutarları, lastik durumu ve yakın zamanda yapılacak bakım kalemleri üzerinden indirim isteyebilirsin.",
                "Piyasa emsallerini inceleyip ilan fiyatının az veya çok olduğunu tespit ettikten sonra teklif ver."
            ],
            "alternatives_same_segment": [
                "Benzer bütçede, aynı segmentte birkaç farklı marka/modeli karşılaştırmak satın alma kararını güçlendirir.",
                "Örneğin aynı segmentte Japon/Kore alternatifleri, daha düşük masraf profili sunabilir.",
                "Premium beklentin yoksa, aynı bütçeyle daha düşük km’li alternatiflere bakmak mantıklı olabilir."
            ]
        },
        "result": "Premium formatta, segment emsallerine ve kullanım profilinize göre yıllık bakım, yakıt, sigorta ve olası riskler ayrı ayrı değerlendirildi. Çıkan sonuçlar yaklaşık bandlar olup, gerçek değerler araç geçmişi ve kullanıma göre değişebilir. Satın alma kararı vermeden önce ekspertiz, tramer ve bakım kayıtlarını mutlaka birlikte incelemeniz önerilir."
    }


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
            "Aile ve karışık kullanım için uygun bir profil sunabilir."
        ],
        "left_cons": ["Gerçek durum eksper raporu ve bakım geçmişine göre netleşecektir."],
        "right_pros": [f"{right_title} doğru bakımla mantıklı bir alternatif olabilir."],
        "right_cons": ["Masraf ve risk tarafında daha dikkatli inceleme gerektirebilir."],
        "use_cases": {
            "family_use": f"Aile kullanımı için {left_title} biraz daha avantajlı varsayılmıştır.",
            "long_distance": "Her iki araç da düzenli bakım ile uzun yolda kullanılabilir.",
            "city_use": "Şehir içi kullanımda şanzıman tipi ve yakıt tüketimi önemli belirleyicilerdir."
        }
    }


def fallback_otobot(question: str) -> Dict[str, Any]:
    return {
        "answer": "Bütçe, yıllık km, kullanım tipi (şehir içi / uzun yol) ve beklentilere göre segment seçmek en mantıklısıdır. Yıllık km yüksekse ekonomik dizel veya LPG’li, düşükse benzinli ya da hybrid bir C segment sedan/SUV iyi başlangıç noktası olabilir. Satın almadan önce ekspertiz ve tramer sorgusunu mutlaka yaptır.",
        "suggested_segments": ["C-sedan", "C-SUV", "B-SUV"],
        "example_models": ["Toyota Corolla", "Renault Megane", "Honda Civic", "Hyundai Tucson"]
    }


# ---------------------------------------------------------
# LLM JSON çağrısı (hata olursa fallback)
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
        resp = client.chat.completions.create(
            model=model_name,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        content = resp.choices[0].message.content
        if isinstance(content, str):
            return json.loads(content)
        return content  # type: ignore[return-value]
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

Sana sağlanan JSON içinde özellikle şu alanları kullan:
- segment: code, name, notes
- market: resale_speed, parts_availability, insurance_level
- costs: listed_price_try, maintenance_yearly_try_min/max, maintenance_routine_yearly_est, maintenance_risk_reserve_yearly_est, yearly_fuel_tr_min/max/mid, insurance_band_tr, consumption_l_per_100km_band
- risk: baseline_risk_level, risk_notes, fuel_risk_comment, age, mileage_km
- profile: yearly_km, yearly_km_band, usage, fuel_preference
- info_quality: level, missing_fields
- anchors_used: chronic listesi vb.

Bilgi seviyesi:
- info_quality.level 'düşük' ise: result metninin başında "Bilgi seviyesi düşük, bazı alanlar eksik" tarzı net bir uyarı ver ve tahminleri genel tut.
- 'orta' veya 'yüksek' ise daha cesur ama yine de makul tahminler yap.

Masraf tahminleri:
- Bakım maliyeti için mutlaka "maintenance_yearly_try_min" ve "maintenance_yearly_try_max" bandını kullan; bunun dışına çıkma.
- cost_estimates.yearly_maintenance_tr değeri bu bandın içinde bir değerde olmalı.
- Yakıt için "yearly_fuel_tr_min/max" bandını kullan; cost_estimates.yearly_fuel_tr bu bandın içinde olsun.
- insurance_band_tr varsa, sigorta/kasko için yaklaşık alt-üst bandı metin içinde "… TL bandı" şeklinde özetle.
- Uçuk rakam yazma; bandlar dışına kesinlikle çıkma.

Kişiselleştirme:
- profile.yearly_km_band, usage ve fuel_preference bilgilerine göre:
  - Yıllık km düşükse: dizel/premium masrafların gereksiz olabileceğini belirt.
  - Yıllık km yüksekse: ekonomik motor ve yakıt sisteminin avantaj/dezavantajlarını anlat.
  - usage 'city' ise: DPF, otomatik şanzıman, yakıt tüketimi şehir içi yönünden değerlendir.
  - usage 'highway' ise: uzun yol konforu, sabit hızda tüketim, koltuk/konfor özelliklerini öne çıkar.
- İlan veya profil tarafında 'öğrenci', 'ilk araç', 'aile aracı', 'çocuk' vb. ifadeler görürsen kişisel yorum ekle (sigorta primi, bagaj, güvenlik vb).

Anchors:
- anchors_used içindeki chronic maddelerini risk_analysis.chronic_issues içine özetleyip doldur.
- risk_analysis.risk_level alanını, risk.baseline_risk_level + kendi değerlendirmene göre belirle.

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

Kurallar:
- Tüm alanları doldur (en azından kısa ama anlamlı cümlelerle).
- "details" içinde:
  - Her başlıkta en az 3 madde bulunsun.
- "result":
  - 20–40 satırlık, başlık + madde karışımı, gerçekten PREMIUM hissettiren, kullanıcıya rehberlik eden bir açıklama olsun.
  - Önce bilgi seviyesi (iyiyse kısaca belirt, düşükse özellikle uyar).
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
    return call_llm_json(
        model_name=OPENAI_MODEL_PREMIUM,
        system_prompt=SYSTEM_PROMPT_PREMIUM,
        user_content=user_content,
        mode="premium",
        req=req,
    )


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
