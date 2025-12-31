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
# ENV + OpenAI
# ---------------------------------------------------------
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

CURRENT_YEAR = int(os.getenv("CURRENT_YEAR", "2025"))

OPENAI_MODEL_DEFAULT = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_MODEL_NORMAL = os.getenv("OPENAI_MODEL_NORMAL", OPENAI_MODEL_DEFAULT)
OPENAI_MODEL_PREMIUM = os.getenv("OPENAI_MODEL_PREMIUM", OPENAI_MODEL_DEFAULT)
OPENAI_MODEL_COMPARE = os.getenv("OPENAI_MODEL_COMPARE", OPENAI_MODEL_DEFAULT)
OPENAI_MODEL_OTOBOT = os.getenv("OPENAI_MODEL_OTOBOT", OPENAI_MODEL_DEFAULT)

# ---------------------------------------------------------
# FastAPI
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
    fuel: Optional[str] = None

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
# Helpers
# ---------------------------------------------------------
def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("ı", "i").replace("ğ", "g").replace("ş", "s").replace("ö", "o").replace("ü", "u").replace("ç", "c")
    s = re.sub(r"\s+", " ", s)
    return s

def _digits(s: str) -> Optional[int]:
    if not s:
        return None
    d = re.sub(r"[^\d]", "", str(s))
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
        v = _digits(txt)
        if v and v > 10000:
            return v
    except:
        pass

    # alternatif alan
    try:
        txt2 = (req.context or {}).get("price") or ""
        v2 = _digits(txt2)
        if v2 and v2 > 10000:
            return v2
    except:
        pass

    return None

def _parse_tramer_amount(req: AnalyzeRequest) -> Optional[int]:
    """
    İlan metninde / context'te 'tramer/hasar/kaza' yakınındaki TL tutarını yakalamaya çalışır.
    Örn: "21.000 TL tramer", "Hasar kaydı 35000" vb.
    """
    hay = []
    if req.ad_description:
        hay.append(req.ad_description)
    ctx = req.context or {}
    for k in ["tramer_text", "damage_text", "hasar_text", "note", "notes"]:
        if ctx.get(k):
            hay.append(str(ctx.get(k)))

    big = " | ".join(hay)
    if not big.strip():
        return None

    # 1) "tramer ... 21.000" veya "hasar ... 35000"
    pat1 = re.compile(r"(tramer|hasar|kaza)[^0-9]{0,25}([0-9\.\,]{3,})", re.IGNORECASE)
    m = pat1.search(big)
    if m:
        v = _digits(m.group(2))
        if v and v >= 1000:
            return v

    # 2) "21.000 TL tramer"
    pat2 = re.compile(r"([0-9\.\,]{3,})\s*(tl|₺)\s*(tramer|hasar|kaza)", re.IGNORECASE)
    m2 = pat2.search(big)
    if m2:
        v = _digits(m2.group(1))
        if v and v >= 1000:
            return v

    return None

# ---------------------------------------------------------
# Segment profilleri
# ---------------------------------------------------------
SEGMENT_PROFILES: Dict[str, Dict[str, Any]] = {
    "B_HATCH": {"name": "B segment (küçük hatchback)", "maintenance_yearly_range": (12000, 25000),
                "insurance_level": "orta", "parts": "çok kolay", "resale": "hızlı"},
    "C_SEDAN": {"name": "C segment (aile sedan/hatchback)", "maintenance_yearly_range": (15000, 32000),
                "insurance_level": "orta", "parts": "kolay", "resale": "hızlı-orta"},
    "C_SUV": {"name": "C segment SUV", "maintenance_yearly_range": (18000, 38000),
              "insurance_level": "orta-yüksek", "parts": "orta-kolay", "resale": "orta-hızlı"},
    "D_SEDAN": {"name": "D segment (konfor sedan)", "maintenance_yearly_range": (22000, 50000),
                "insurance_level": "orta-yüksek", "parts": "orta", "resale": "orta"},
    "PREMIUM_D": {"name": "Premium D segment", "maintenance_yearly_range": (32000, 75000),
                  "insurance_level": "yüksek", "parts": "orta-zor", "resale": "orta"},
    "E_SEGMENT": {"name": "E segment / üst sınıf", "maintenance_yearly_range": (45000, 120000),
                  "insurance_level": "çok yüksek", "parts": "zor", "resale": "yavaş-orta"},
}

ANCHORS: List[Dict[str, Any]] = [
    {"key": "honda civic", "segment": "C_SEDAN", "aliases": ["civic"],
     "chronic": ["LPG takılı Civiclerde subap ve ayar takibi önemli"], "parts": "kolay", "resale": "hızlı"},
    {"key": "audi a4", "segment": "PREMIUM_D", "aliases": ["a4", "a4 allroad", "allroad"],
     "chronic": ["S-tronic/DSG geçmişi önemli", "Baz TFSI nesillerinde yağ tüketimi geçmişi sorgulanmalı"],
     "parts": "orta-zor", "resale": "orta"},
    {"key": "bmw 320", "segment": "PREMIUM_D", "aliases": ["320i", "320d", "3.20", "3.20i", "3.20d", "3 series", "3 serisi"],
     "chronic": ["Burç/ön takım zamanla yıpranabilir", "Turbo-dizelde EGR/DPF ve zincir/yağ bakımı ihmal edilmemeli"],
     "parts": "orta-zor", "resale": "orta"},
    {"key": "mercedes c200", "segment": "PREMIUM_D", "aliases": ["c180", "c200", "c220", "c class", "c serisi"],
     "chronic": ["Elektronik/konfor donanımlarında yaşa bağlı arızalar", "Bakım standardı kritik"],
     "parts": "orta", "resale": "orta"},
]

def detect_segment(make: str, model: str) -> str:
    s = _norm(f"{make} {model}")
    if any(k in s for k in ["bmw", "mercedes", "audi", "volvo", "lexus", "range rover"]):
        return "PREMIUM_D"
    if any(k in s for k in ["clio", "polo", "i20", "corsa", "yaris", "fiesta", "fabia"]):
        return "B_HATCH"
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
        key = _norm(a.get("key", ""))
        if key and key in target:
            score += 10
        for al in (a.get("aliases") or []):
            if _norm(al) in target:
                score += 8
        if score > 0:
            scored.append((score, a))
    scored.sort(key=lambda x: x[0], reverse=True)
    if scored:
        return [x[1] for x in scored[:limit]]
    return [a for a in ANCHORS if a.get("segment") == segment][:limit]

# ---------------------------------------------------------
# Info quality
# ---------------------------------------------------------
def evaluate_info_quality(req: AnalyzeRequest) -> Dict[str, Any]:
    v = req.vehicle
    p = req.profile or Profile()
    missing: List[str] = []

    if not (v.make and v.make.strip()): missing.append("marka")
    if not (v.model and v.model.strip()): missing.append("model")
    if v.year is None: missing.append("yil")
    if v.mileage_km is None: missing.append("km")
    if not v.fuel: missing.append("yakit")

    ad_text = (req.ad_description or "").strip()
    if len(ad_text) < 40: missing.append("ilan_aciklamasi_kisa")

    if p.usage not in ("city", "mixed", "highway"):
        missing.append("profil_kullanim_tipi")

    if len(missing) <= 1 and len(ad_text) >= 120:
        level = "yüksek"
    elif len(missing) <= 3 and len(ad_text) >= 40:
        level = "orta"
    else:
        level = "düşük"

    return {"level": level, "missing_fields": missing, "ad_length": len(ad_text)}

# ---------------------------------------------------------
# Cost + risk + pazarlık
# ---------------------------------------------------------
def estimate_costs(req: AnalyzeRequest) -> Dict[str, Any]:
    v = req.vehicle
    p = req.profile or Profile()

    segment_code = detect_segment(v.make, v.model)
    seg = SEGMENT_PROFILES.get(segment_code, SEGMENT_PROFILES["C_SEDAN"])

    listed_price = _parse_listed_price(req)
    tramer_amount = _parse_tramer_amount(req)

    age = None
    if v.year:
        age = max(0, CURRENT_YEAR - v.year)
    mileage = int(v.mileage_km or 0)

    base_min, base_max = seg["maintenance_yearly_range"]

    # çarpanlar
    age_mult = 1.0
    if age is not None:
        if age >= 15: age_mult = 1.8
        elif age >= 10: age_mult = 1.4
        elif age >= 6: age_mult = 1.15

    km_mult = 1.0
    if mileage >= 250_000: km_mult = 1.8
    elif mileage >= 180_000: km_mult = 1.4
    elif mileage >= 120_000: km_mult = 1.2

    usage_mult = 1.0
    if p.usage == "city": usage_mult = 1.15
    elif p.usage == "highway": usage_mult = 0.95

    fuel = (v.fuel or p.fuel_preference or "").lower()
    fuel_mult_maint = 1.0
    fuel_risk = "orta"

    if fuel == "diesel":
        fuel_mult_maint = 1.05
        if p.usage == "city" and mileage >= 120_000:
            fuel_risk = "orta-yüksek (DPF/EGR/enjektör riski şehir içiyle artabilir)"
    elif fuel == "lpg":
        fuel_mult_maint = 0.95
        fuel_risk = "orta (montaj/ayar kalitesi kritik, subap takibi şart)"
    elif fuel in ("hybrid", "electric"):
        fuel_mult_maint = 0.9
        fuel_risk = "düşük-orta (batarya sağlığına bağlı)"

    maint_min = int(base_min * age_mult * km_mult * usage_mult * fuel_mult_maint)
    maint_max = int(base_max * age_mult * km_mult * usage_mult * fuel_mult_maint)

    mid_maint = int((maint_min + maint_max) / 2)
    routine_est = int(mid_maint * 0.65)
    risk_reserve_est = mid_maint - routine_est

    # yakıt bandı
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
    if fuel == "diesel": fuel_mult_cost = 0.9
    elif fuel == "lpg": fuel_mult_cost = 0.75
    elif fuel in ("hybrid", "electric"): fuel_mult_cost = 0.7

    fuel_min = int(fuel_base[0] * km_factor * fuel_mult_cost)
    fuel_max = int(fuel_base[1] * km_factor * fuel_mult_cost)
    fuel_mid = int((fuel_min + fuel_max) / 2)

    # sigorta/kasko
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
        ins_min = max(int(listed_price * ir_min), 5000)
        ins_max = min(int(listed_price * ir_max), 160000)

    # risk seviyesi
    risk_level = "orta"
    risk_notes: List[str] = []
    if age is not None and age >= 15:
        risk_level = "yüksek"; risk_notes.append("Araç yaşı yüksek; büyük masraf ihtimali artmış olabilir.")
    elif age is not None and age >= 10:
        risk_level = "orta-yüksek"; risk_notes.append("Yaş nedeniyle büyük bakım ihtimali çıkabilir.")

    if mileage >= 250_000:
        risk_level = "yüksek"; risk_notes.append("Km çok yüksek; motor/şanzıman revizyon riski artabilir.")
    elif mileage >= 180_000 and risk_level != "yüksek":
        risk_level = "orta-yüksek"; risk_notes.append("Km yüksek; yürüyen aksam ve mekanik masraf ihtimali artar.")
    elif mileage >= 120_000 and risk_level == "orta":
        risk_notes.append("Km 120 bin+ bandında; büyük bakım kalemleri yaklaşıyor olabilir.")

    if "yüksek" in fuel_risk:
        risk_level = "orta-yüksek" if risk_level == "orta" else risk_level

    # pazarlık önerisi (fiyat varsa TL band, yoksa %)
    suggested_discount_pct = 0.04  # minimum pazarlık payı
    if mileage >= 180_000: suggested_discount_pct += 0.05
    elif mileage >= 120_000: suggested_discount_pct += 0.03

    if age is not None and age >= 10: suggested_discount_pct += 0.03
    elif age is not None and age >= 6: suggested_discount_pct += 0.015

    if tramer_amount:
        if tramer_amount >= 50000: suggested_discount_pct += 0.06
        elif tramer_amount >= 20000: suggested_discount_pct += 0.03
        else: suggested_discount_pct += 0.015

    suggested_discount_pct = min(0.18, suggested_discount_pct)

    offer_band = None
    if listed_price:
        offer_min = int(listed_price * (1 - suggested_discount_pct - 0.03))
        offer_max = int(listed_price * (1 - max(0.0, suggested_discount_pct - 0.02)))
        offer_band = [max(0, offer_min), max(0, offer_max)]

    return {
        "segment_code": segment_code,
        "segment_name": seg["name"],
        "age": age,
        "mileage_km": mileage,
        "listed_price_try": listed_price,
        "tramer_amount_try": tramer_amount,

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

        "suggested_discount_pct": round(suggested_discount_pct * 100, 1),
        "offer_band_try": offer_band,

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
    if p.yearly_km <= 8000: yearly_km_band = "düşük"
    elif p.yearly_km >= 30000: yearly_km_band = "yüksek"

    return {
        "segment": {"code": seg_info["segment_code"], "name": seg_info["segment_name"]},
        "market": {"resale_speed": seg_info["resale_speed"], "parts_availability": seg_info["parts_availability"],
                   "insurance_level": seg_info["insurance_level"]},
        "costs": {
            "listed_price_try": seg_info["listed_price_try"],
            "tramer_amount_try": seg_info["tramer_amount_try"],
            "maintenance_yearly_try_min": seg_info["maintenance_yearly_try_min"],
            "maintenance_yearly_try_max": seg_info["maintenance_yearly_try_max"],
            "maintenance_routine_yearly_est": seg_info["maintenance_routine_yearly_est"],
            "maintenance_risk_reserve_yearly_est": seg_info["maintenance_risk_reserve_yearly_est"],
            "yearly_fuel_tr_min": seg_info["yearly_fuel_tr_min"],
            "yearly_fuel_tr_max": seg_info["yearly_fuel_tr_max"],
            "yearly_fuel_tr_mid": seg_info["yearly_fuel_tr_mid"],
            "insurance_band_tr": seg_info["insurance_band_tr"],
            "consumption_l_per_100km_band": seg_info["consumption_l_per_100km_band"],
            "suggested_discount_pct": seg_info["suggested_discount_pct"],
            "offer_band_try": seg_info["offer_band_try"],
        },
        "risk": {
            "fuel_risk_comment": seg_info["fuel_risk_comment"],
            "baseline_risk_level": seg_info["risk_level"],
            "risk_notes": seg_info["risk_notes"],
            "age": seg_info["age"],
            "mileage_km": seg_info["mileage_km"],
        },
        "profile": {"yearly_km": p.yearly_km, "yearly_km_band": yearly_km_band,
                    "usage": p.usage, "fuel_preference": p.fuel_preference},
        "info_quality": info_q,
        "anchors_used": [
            {"key": a.get("key"), "segment": a.get("segment"),
             "chronic": a.get("chronic", []), "parts": a.get("parts"), "resale": a.get("resale")}
            for a in anchors
        ],
    }

# ---------------------------------------------------------
# User content (LLM'e net sayılarla brief)
# ---------------------------------------------------------
def build_user_content(req: AnalyzeRequest, mode: str) -> str:
    v = req.vehicle or Vehicle()
    p = req.profile or Profile()
    ad_text = (req.ad_description or "").strip()
    enriched = build_enriched_context(req)

    # Screenshot sayısı bilgisi (OCR yok ama “var” bilgisini veriyoruz)
    all_ss: List[str] = []
    if req.screenshot_base64:
        all_ss.append(req.screenshot_base64)
    if req.screenshots_base64:
        all_ss.extend([s for s in req.screenshots_base64 if s])

    ss_info = ""
    if all_ss:
        ss_info = (
            f"\nKullanıcı {len(all_ss)} adet ekran görüntüsü ekledi. "
            "Eğer kullanıcı metninde yoksa; görüntülerden emin olmadığın donanımı ASLA uydurma, 'belirsiz' de."
        )

    # Net brief (model kaçamasın)
    c = enriched["costs"]
    r = enriched["risk"]
    mk = enriched["market"]
    iq = enriched["info_quality"]

    brief = f"""
NET VERİLER (BUNLARI RESULT İÇİNDE MUTLAKA TL BANDIYLA YAZ):
- Segment: {enriched['segment']['name']} / {enriched['segment']['code']}
- Yaş: {r.get('age')} yıl, Km: {r.get('mileage_km')}
- Yıllık bakım bandı: {c['maintenance_yearly_try_min']}–{c['maintenance_yearly_try_max']} TL
  - rutin pay: {c['maintenance_routine_yearly_est']} TL, risk payı: {c['maintenance_risk_reserve_yearly_est']} TL
- Yıllık yakıt bandı: {c['yearly_fuel_tr_min']}–{c['yearly_fuel_tr_max']} TL (orta: {c['yearly_fuel_tr_mid']} TL)
- Sigorta/Kasko bandı: {c['insurance_band_tr'] if c['insurance_band_tr'] else 'Fiyat yoksa %2.5–%6.5 araç bedeli bandı yaz'}
- Tramer: {c.get('tramer_amount_try') if c.get('tramer_amount_try') else 'belirsiz'}
- Pazarlık önerisi: %{c.get('suggested_discount_pct')} indirim; teklif bandı: {c.get('offer_band_try') if c.get('offer_band_try') else 'Fiyat yoksa % bandı ver'}
- 2.el hızı: {mk['resale_speed']}, parça bulunabilirliği: {mk['parts_availability']}, sigorta seviyesi: {mk['insurance_level']}
- Bilgi seviyesi: {iq['level']} (eksikler: {iq['missing_fields']})
""".strip()

    if not (v.make.strip() or v.model.strip() or ad_text or all_ss):
        ad_text = "Kullanıcı çok az bilgi verdi. Türkiye ikinci el piyasasında genel kriterlerle, tahminleri band içinde tutarak değerlendirme yap."

    return f"""
Kullanıcı Oto Analiz uygulamasında **{mode}** modunda analiz istiyor.

Araç:
- Marka: {v.make or "-"}
- Model: {v.model or "-"}
- Yıl: {v.year or "-"}
- Km: {v.mileage_km or "-"}
- Yakıt: {v.fuel or p.fuel_preference}

Profil:
- Yıllık km: {p.yearly_km}
- Kullanım: {p.usage}
- Yakıt tercihi: {p.fuel_preference}

İlan açıklaması / not:
{ad_text if ad_text else "-"}

--- ENRICHED JSON (referans) ---
{json.dumps(enriched, ensure_ascii=False)}
--- NET BRIEF ---
{brief}
{ss_info}
""".strip()

# ---------------------------------------------------------
# Premium kalite kontrol (model kaçarsa 1 kez yeniden yazdır)
# ---------------------------------------------------------
def premium_needs_retry(obj: Dict[str, Any]) -> Tuple[bool, List[str]]:
    reasons = []
    try:
        ce = obj.get("cost_estimates", {})
        res = (obj.get("result") or "").strip()

        # result kısa mı?
        if len(res) < 900:
            reasons.append("result çok kısa (<900 char)")

        # masraf alanları var mı?
        for k in ["yearly_maintenance_tr_min", "yearly_maintenance_tr_max", "yearly_fuel_tr_min", "yearly_fuel_tr_max"]:
            if ce.get(k) in (None, 0):
                reasons.append(f"cost_estimates.{k} boş/0")

        # result içinde TL bandı geçiyor mu?
        must_words = ["Bakım", "Yakıt", "Sigorta", "Kasko", "TL", "Pazarlık"]
        missing = [w for w in must_words if w.lower() not in res.lower()]
        if missing:
            reasons.append(f"result içinde eksik başlıklar: {missing}")

        # details her başlık min 3 mü?
        details = obj.get("details", {})
        for sec in ["personal_fit","maintenance_breakdown","insurance_kasko","parts_and_service","resale_market","negotiation_tips","alternatives_same_segment"]:
            arr = details.get(sec, [])
            if not isinstance(arr, list) or len(arr) < 3:
                reasons.append(f"details.{sec} < 3 madde")
    except Exception as e:
        reasons.append(f"validation exception: {e}")
    return (len(reasons) > 0), reasons

# ---------------------------------------------------------
# LLM JSON call (premium retry)
# ---------------------------------------------------------
def call_llm_json(model_name: str, system_prompt: str, user_content: str, mode: str, req: Any) -> Dict[str, Any]:
    if client is None:
        return {"error": "OPENAI_API_KEY yok"}  # burada istersen fallback ekleyebilirsin

    def _call(sp: str, uc: str) -> Dict[str, Any]:
        resp = client.chat.completions.create(
            model=model_name,
            response_format={"type": "json_object"},
            temperature=0.35,
            max_tokens=2200,
            messages=[
                {"role": "system", "content": sp},
                {"role": "user", "content": uc},
            ],
        )
        content = resp.choices[0].message.content
        return json.loads(content) if isinstance(content, str) else content

    try:
        out = _call(system_prompt, user_content)

        if mode == "premium":
            need, reasons = premium_needs_retry(out)
            if need:
                repair_prompt = SYSTEM_PROMPT_PREMIUM_REPAIR
                repair_user = json.dumps({
                    "reasons": reasons,
                    "previous_json": out
                }, ensure_ascii=False)
                out2 = _call(repair_prompt, repair_user)
                return out2

        return out
    except Exception as e:
        print(f"LLM hatası ({mode}): {e}")
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

# ---------------------------------------------------------
# System prompts
# ---------------------------------------------------------
SYSTEM_PROMPT_NORMAL = """
Sen 'Oto Analiz' uygulaması için çalışan bir araç ilanı analiz asistanısın.
ÇIKTIYI SADECE GEÇERLİ JSON OLARAK DÖN.

Kurallar:
- Donanım / özellik uydurma. İlanda/sunumda yoksa "belirsiz" de.
- PREVIEW'de 'alınır/alınmaz/sakın/tehlikeli' yok.
- Fiyat rakamı PREVIEW'e yazma (etiket: Uygun/Normal/Yüksek ya da null).
- Dil: Türkçe.
JSON şablonu:
{
  "scores":{"overall_100":0,"mechanical_100":0,"body_100":0,"economy_100":0},
  "summary":{"short_comment":"","pros":[],"cons":[],"estimated_risk_level":"orta"},
  "preview":{"title":"","price_tag":null,"spoiler":"","bullets":[]},
  "result":""
}
"""

SYSTEM_PROMPT_PREMIUM = """
Sen 'Oto Analiz' uygulamasının PREMIUM analiz asistanısın.
ÇIKTIYI SADECE GEÇERLİ JSON OLARAK DÖN.

KRİTİK KURALLAR:
- Donanım/opsiyon UYDURMA. İlanda yoksa "belirsiz" yaz.
- RESULT içinde MUTLAKA şu başlıklar olacak ve TL bandı yazılacak:
  1) Masraf Tahmini (Bakım TL min-max, Yakıt TL min-max, Sigorta/Kasko TL bandı veya % bandı)
  2) Kronik & Kontrol Listesi (anchors_used + dizel/lpg riskleri)
  3) Piyasa/Fiyat & Pazarlık (suggested_discount_pct ve offer_band_try)
  4) Kullanıcıya Uygunluk (profile.yearly_km_band + usage)
  5) 2. El / Likidite / Parça
- RESULT min 20 satır hissi verecek kadar uzun olsun (kısa geçme).
- details içindeki her başlıkta EN AZ 3 madde olacak.

JSON ŞABLONU:
{
  "scores":{"overall_100":0,"mechanical_100":0,"body_100":0,"economy_100":0,"comfort_100":0,"family_use_100":0,"resale_100":0},
  "cost_estimates":{
    "yearly_maintenance_tr":0,"yearly_maintenance_tr_min":0,"yearly_maintenance_tr_max":0,
    "yearly_fuel_tr":0,"yearly_fuel_tr_min":0,"yearly_fuel_tr_max":0,
    "insurance_level":"orta","insurance_band_tr":null,"notes":""
  },
  "risk_analysis":{"chronic_issues":[],"risk_level":"orta","warnings":[]},
  "summary":{"short_comment":"","pros":[],"cons":[],"who_should_buy":""},
  "preview":{"title":"","price_tag":null,"spoiler":"","bullets":[]},
  "details":{
    "personal_fit":[],"maintenance_breakdown":[],"insurance_kasko":[],"parts_and_service":[],
    "resale_market":[],"negotiation_tips":[],"alternatives_same_segment":[]
  },
  "result":""
}

NOT: cost_estimates içindeki sayıları kullanıcıya da RESULT içinde yaz.
"""

SYSTEM_PROMPT_PREMIUM_REPAIR = """
Sen bir JSON kalite düzelticisisin.
Sana 'previous_json' ve 'reasons' verilecek.
Görev: previous_json'u KURALLARA UYGUN hale getir ve SADECE DÜZELTİLMİŞ JSON döndür.

Kurallar:
- Donanım uydurma.
- details her başlık min 3 madde.
- result uzun + başlıklı + TL bandları içerir (Bakım/Yakıt/Sigorta/Kasko/Pazarlık).
- cost_estimates min-max alanları 0/None olamaz (brief'e göre doldur).
"""

SYSTEM_PROMPT_COMPARE = """
Sadece JSON döndür:
{
  "better_overall":"left",
  "summary":"",
  "left_pros":[],
  "left_cons":[],
  "right_pros":[],
  "right_cons":[],
  "use_cases":{"family_use":"","long_distance":"","city_use":""}
}
Dil: Türkçe. Donanım uydurma.
"""

SYSTEM_PROMPT_OTOBOT = """
Çıktı sadece JSON:
{"answer":"","suggested_segments":[],"example_models":[]}
Dil: Türkçe.
"""

# ---------------------------------------------------------
# Healthcheck
# ---------------------------------------------------------
@app.get("/")
async def root() -> Dict[str, Any]:
    return {"ok": True, "message": "Oto Analiz backend çalışıyor."}

# ---------------------------------------------------------
# NORMAL
# ---------------------------------------------------------
@app.post("/analyze")
async def analyze(req: AnalyzeRequest) -> Dict[str, Any]:
    user_content = build_user_content(req, mode="normal")
    return call_llm_json(OPENAI_MODEL_NORMAL, SYSTEM_PROMPT_NORMAL, user_content, "normal", req)

# ---------------------------------------------------------
# PREMIUM
# ---------------------------------------------------------
@app.post("/premium_analyze")
async def premium_analyze(req: AnalyzeRequest) -> Dict[str, Any]:
    user_content = build_user_content(req, mode="premium")
    return call_llm_json(OPENAI_MODEL_PREMIUM, SYSTEM_PROMPT_PREMIUM, user_content, "premium", req)

# ---------------------------------------------------------
# COMPARE
# ---------------------------------------------------------
@app.post("/compare_analyze")
async def compare_analyze(req: CompareRequest) -> Dict[str, Any]:
    lv = req.left.vehicle
    rv = req.right.vehicle
    profile = req.profile or Profile()

    user_content = f"""
Sol: {lv.make} {lv.model} ({lv.year}) {lv.mileage_km}km {lv.fuel}
Sol açıklama: {req.left.ad_description or "-"}

Sağ: {rv.make} {rv.model} ({rv.year}) {rv.mileage_km}km {rv.fuel}
Sağ açıklama: {req.right.ad_description or "-"}

Profil: yıllık {profile.yearly_km}km, kullanım {profile.usage}, yakıt tercih {profile.fuel_preference}
""".strip()

    return call_llm_json(OPENAI_MODEL_COMPARE, SYSTEM_PROMPT_COMPARE, user_content, "compare", req)

# ---------------------------------------------------------
# OTOBOT
# ---------------------------------------------------------
@app.post("/otobot")
async def otobot(req: OtoBotRequest) -> Dict[str, Any]:
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Soru boş olamaz.")
    return call_llm_json(OPENAI_MODEL_OTOBOT, SYSTEM_PROMPT_OTOBOT, question, "otobot", req)
