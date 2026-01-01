import os
import json
import re
import math
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


# =========================================================
# ENV + CLIENT
# =========================================================
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = None
if api_key and OpenAI is not None:
    client = OpenAI(api_key=api_key)

OPENAI_MODEL_DEFAULT = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_MODEL_NORMAL = os.getenv("OPENAI_MODEL_NORMAL", OPENAI_MODEL_DEFAULT)
OPENAI_MODEL_PREMIUM = os.getenv("OPENAI_MODEL_PREMIUM", OPENAI_MODEL_DEFAULT)
OPENAI_MODEL_COMPARE = os.getenv("OPENAI_MODEL_COMPARE", OPENAI_MODEL_DEFAULT)
OPENAI_MODEL_OTOBOT = os.getenv("OPENAI_MODEL_OTOBOT", OPENAI_MODEL_DEFAULT)

DATA_DIR = os.getenv("DATA_DIR", "data")  # bundle içinde ./data


# =========================================================
# APP
# =========================================================
app = FastAPI(title="Oto Analiz Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# MODELS
# =========================================================
class Profile(BaseModel):
    # mevcut alanlar (Flutter tarafını kırmamak için aynı)
    yearly_km: int = Field(15000, ge=0, le=100000)
    usage: str = "mixed"              # city / mixed / highway
    fuel_preference: str = "gasoline" # gasoline / diesel / lpg / hybrid / electric

    # yeni alanlar (opsiyonel - göndermesen de olur)
    family: str = "couple"            # single / couple / kids
    budget_sensitivity: str = "medium"  # low / medium / high
    priority: str = "balance"           # comfort / performance / balance

    class Config:
        extra = "allow"  # ileride yeni alan ekleyince 422 olmasın


class Vehicle(BaseModel):
    make: str = ""
    model: str = ""
    year: Optional[int] = Field(None, ge=1980, le=2035)
    mileage_km: Optional[int] = Field(None, ge=0)
    fuel: Optional[str] = None        # gasoline / diesel / lpg / hybrid / electric
    transmission: Optional[str] = None  # manual / auto / dsg / cvt / torque / unknown

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


# =========================================================
# BASIC HELPERS
# =========================================================
BANNED_WORDS = [
    "alınır", "alinir", "alınmaz", "alinmaz",
    "tehlikeli", "riskli", "sakın", "sakin",
]

def _contains_banned(text: str) -> bool:
    t = (text or "").lower()
    return any(w in t for w in BANNED_WORDS)

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


def _fmt_try(n: Optional[int]) -> str:
    if n is None:
        return "-"
    return f"{int(n):,}".replace(",", ".")


def _clamp(n: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, n))


def _clamp_int(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, n))


def _parse_listed_price(req: AnalyzeRequest) -> Optional[int]:
    try:
        txt = (req.context or {}).get("listed_price_text") or ""
        v = _digits(str(txt))
        if v and v > 10_000:
            return v
    except:
        pass

    try:
        txt2 = getattr(req, "text", None)
        if isinstance(txt2, str):
            v2 = _digits(txt2)
            if v2 and v2 > 10_000:
                return v2
    except:
        pass

    return None


def _infer_engine_cc_from_text(text: str) -> Optional[int]:
    if not text:
        return None
    t = text.lower().replace(",", ".")
    m = re.search(r"(?<!\d)(0\.\d|1\.\d|2\.\d|3\.\d|4\.\d)(?!\d)", t)
    if not m:
        return None
    try:
        liters = float(m.group(1))
        cc = int(round(liters * 1000))
        if 600 <= cc <= 6000:
            return cc
    except:
        pass
    return None


def _guess_tags(req: AnalyzeRequest) -> List[str]:
    blob = f"{req.vehicle.make} {req.vehicle.model} {req.ad_description or ''} {json.dumps(req.context or {}, ensure_ascii=False)}"
    t = _norm(blob)
    tags: List[str] = []

    if "dsg" in t or "s tronic" in t or "s-tronic" in t:
        tags.append("dsg")
    if "dct" in t or "edc" in t or "powershift" in t:
        tags.append("dct_optional")
    if "cvt" in t:
        tags.append("cvt")
    if "lpg" in t or "prins" in t or "landi" in t:
        tags.append("lpg_common")
    if "hybrid" in t:
        tags.append("hybrid")
    if "electric" in t or "ev" in t or "elektrik" in t:
        tags.append("ev")
    if "4x4" in t or "awd" in t or "quattro" in t:
        tags.append("awd_optional")
    if "turbo" in t or "tce" in t or "tsi" in t or "ecoboost" in t:
        tags.append("turbo_small")

    f = (req.vehicle.fuel or req.profile.fuel_preference or "").lower().strip()
    if f == "diesel":
        tags.append("diesel")
    if f == "lpg":
        tags.append("lpg_common")
    if f == "hybrid":
        tags.append("hybrid")
    if f == "electric":
        tags.append("ev")

    out = []
    for x in tags:
        if x not in out:
            out.append(x)
    return out


# =========================================================
# DATA PACK LOADER
# =========================================================
def _load_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return default


def _dp(path_in_data: str) -> str:
    return os.path.join(DATA_DIR, path_in_data)


ANCHORS: List[Dict[str, Any]] = _load_json(_dp("anchors_tr_popular_96.json"), [])
VEHICLE_PROFILES: List[Dict[str, Any]] = _load_json(_dp("vehicle_profiles_96_v1.json"), [])
TRAFFIC_CAPS: Dict[str, Any] = _load_json(_dp("traffic_caps_tr_2025_12_seed.json"), {})
MTV_PACK: Dict[str, Any] = _load_json(_dp("mtv_tr_2025_2026_estimated_1895.json"), {})
FIXED_COSTS: Dict[str, Any] = _load_json(_dp("fixed_costs_tr_2026_estimated.json"), {})


def _build_segment_stats(profiles: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    agg: Dict[str, Dict[str, List[int]]] = {}
    for p in profiles:
        seg = p.get("segment") or "C_SEDAN"
        idx = (p.get("indices") or {})
        if seg not in agg:
            agg[seg] = {
                "parts_availability_score_1_5": [],
                "parts_cost_index_1_5": [],
                "service_network_index_1_5": [],
                "resale_liquidity_score_1_5": [],
            }
        for k in agg[seg].keys():
            v = idx.get(k)
            if isinstance(v, int):
                agg[seg][k].append(v)

    stats: Dict[str, Dict[str, float]] = {}
    for seg, cols in agg.items():
        stats[seg] = {}
        for k, arr in cols.items():
            if arr:
                stats[seg][k] = sum(arr) / max(1, len(arr))
            else:
                stats[seg][k] = 3.0
    return stats


SEGMENT_STATS = _build_segment_stats(VEHICLE_PROFILES)


# =========================================================
# SEGMENT PROFILES (bandlar)
# =========================================================
SEGMENT_PROFILES: Dict[str, Dict[str, Any]] = {
    "B_HATCH": {
        "name": "B segment (küçük hatchback)",
        "maintenance_yearly_range": (12000, 25000),
        "insurance_level": "orta",
        "notes": [
            "Parça ve usta erişimi genelde kolaydır; şehir içi kullanım için uygundur.",
            "Doğru bakım disipliniyle yıllık masraflar çoğu zaman yönetilebilir kalır.",
        ],
    },
    "C_SEDAN": {
        "name": "C segment (aile sedan/hatchback)",
        "maintenance_yearly_range": (15000, 32000),
        "insurance_level": "orta",
        "notes": [
            "Türkiye’de en likit sınıflardan biridir; temiz örneklerin alıcısı genelde vardır.",
            "Konfor/donanım arttıkça bakım maliyeti B segmente göre bir miktar yükselir.",
        ],
    },
    "C_SUV": {
        "name": "C segment SUV",
        "maintenance_yearly_range": (18000, 38000),
        "insurance_level": "orta-yüksek",
        "notes": [
            "SUV gövdede lastik/fren/alt takım maliyeti kompakt araçlara göre artabilir.",
            "Aile/uzun yol için dengeli; tüketim ve lastik maliyeti biraz daha yüksek olabilir.",
        ],
    },
    "D_SEDAN": {
        "name": "D segment (konfor sedan)",
        "maintenance_yearly_range": (22000, 50000),
        "insurance_level": "orta-yüksek",
        "notes": [
            "Konfor ve donanım yüksek; parça/işçilik maliyetleri C segmente göre artabilir.",
        ],
    },
    "PREMIUM_D": {
        "name": "Premium sınıf",
        "maintenance_yearly_range": (32000, 75000),
        "insurance_level": "yüksek",
        "notes": [
            "Premium sınıfta işçilik ve parça maliyetleri belirgin yükselebilir.",
            "Donanım yoğunluğu arttıkça kontrol noktaları da artar.",
        ],
    },
    "E_SEGMENT": {
        "name": "E segment / üst sınıf",
        "maintenance_yearly_range": (45000, 120000),
        "insurance_level": "çok yüksek",
        "notes": [
            "Üst sınıf büyük gövdeli araçlarda masraf kalemleri belirgin şekilde artabilir.",
        ],
    },
}


# =========================================================
# SEGMENT DETECTION
# =========================================================
def detect_segment(make: str, model: str) -> str:
    s = _norm(f"{make} {model}")

    if any(k in s for k in ["bmw", "mercedes", "audi", "volvo", "lexus", "range rover", "land rover"]):
        return "PREMIUM_D"

    if any(k in s for k in ["clio", "polo", "i20", "corsa", "yaris", "fiesta", "fabia", "ibiza"]):
        return "B_HATCH"
    if any(k in s for k in ["corolla", "civic", "megane", "astra", "focus", "egea", "tipo", "elantra", "i30"]):
        return "C_SEDAN"
    if any(k in s for k in ["qashqai", "tucson", "sportage", "kuga", "3008", "duster"]):
        return "C_SUV"
    if any(k in s for k in ["passat", "superb", "508", "insignia", "camry"]):
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


def _profile_by_key(key: str) -> Optional[Dict[str, Any]]:
    nk = _norm(key)
    for p in VEHICLE_PROFILES:
        if _norm(p.get("key", "")) == nk:
            return p
    return None


def resolve_vehicle_profile(req: AnalyzeRequest, segment: str) -> Dict[str, Any]:
    v = req.vehicle
    target = _norm(f"{v.make} {v.model}")
    anchors = find_anchor_matches(v.make, v.model, segment, limit=3)

    for a in anchors:
        p = _profile_by_key(a.get("key", ""))
        if p:
            return {
                "match": "exact_or_anchor",
                "matched_key": p.get("key"),
                "profile": p,
                "anchors_used": anchors,
            }

    for p in VEHICLE_PROFILES:
        k = _norm(p.get("key", ""))
        if k and k in target:
            return {"match": "substring", "matched_key": p.get("key"), "profile": p, "anchors_used": anchors}
        for al in (p.get("aliases") or []):
            if _norm(al) and _norm(al) in target:
                return {"match": "alias", "matched_key": p.get("key"), "profile": p, "anchors_used": anchors}

    st = SEGMENT_STATS.get(segment, {
        "parts_availability_score_1_5": 3.0,
        "parts_cost_index_1_5": 3.0,
        "service_network_index_1_5": 3.0,
        "resale_liquidity_score_1_5": 3.0,
    })
    avg_profile = {
        "key": None,
        "segment": segment,
        "indices": {k: int(round(vv)) for k, vv in st.items()},
        "risk_patterns": [],
        "big_maintenance_watchlist": [],
        "inspection_checklist": [],
        "tags": _guess_tags(req),
        "disclaimer": "Bu model için doğrudan profil bulunamadı; aynı segment emsallerinden tahmini endeksler üretildi.",
    }
    return {"match": "segment_estimate", "matched_key": None, "profile": avg_profile, "anchors_used": anchors}


# =========================================================
# INFO QUALITY
# =========================================================
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

    return {"level": level, "missing_fields": missing, "ad_length": ad_len}


# =========================================================
# TAX / FIXED (senin kodun olduğu gibi)
# =========================================================
def _age_group_car(age: int) -> str:
    if age <= 3:
        return "1_3"
    if age <= 6:
        return "4_6"
    if age <= 11:
        return "7_11"
    if age <= 15:
        return "12_15"
    return "16_plus"


def estimate_mtv(req: AnalyzeRequest, tax_year: Optional[int] = None) -> Dict[str, Any]:
    pack = MTV_PACK or {}
    if not pack:
        return {"ok": False, "note": "MTV pack bulunamadı."}

    table_container = pack.get("mtv_2026_estimated") or pack.get("mtv_2025_official")
    if not table_container:
        return {"ok": False, "note": "MTV tablo alanı eksik."}

    ty = int(tax_year or (req.context or {}).get("mtv_year") or date.today().year)

    v = req.vehicle or Vehicle()
    reg_year = (req.context or {}).get("registration_year") or v.year
    try:
        reg_year = int(reg_year) if reg_year else None
    except:
        reg_year = None

    engine_cc = (req.context or {}).get("engine_cc")
    if engine_cc is None:
        engine_cc = _infer_engine_cc_from_text((req.ad_description or "") + " " + json.dumps(req.context or {}, ensure_ascii=False))
    try:
        engine_cc = int(engine_cc) if engine_cc else None
    except:
        engine_cc = None

    vehicle_value = (req.context or {}).get("vehicle_value_try") or (req.context or {}).get("kasko_value_try")
    if vehicle_value is None:
        vehicle_value = _parse_listed_price(req)
    try:
        vehicle_value = int(vehicle_value) if vehicle_value else None
    except:
        vehicle_value = None

    age = 0
    if reg_year:
        age = max(0, ty - reg_year)
    elif v.year:
        age = max(0, ty - v.year)

    age_key = _age_group_car(age)
    is_value_based = (reg_year is not None and reg_year >= 2018)

    if is_value_based:
        table = table_container.get("tariff_I_automobile_value_based") or []
    else:
        table = table_container.get("tariff_IA_pre2018_automobile") or []

    segment = detect_segment(v.make, v.model)

    if engine_cc is None:
        seg_cc_band = {
            "B_HATCH": (0, 1600),
            "C_SEDAN": (0, 1600),
            "C_SUV": (1301, 2000),
            "D_SEDAN": (1601, 2000),
            "PREMIUM_D": (1601, 3000),
            "E_SEGMENT": (2001, 99999),
        }.get(segment, (0, 2000))
        cc_candidates = [seg_cc_band[0], seg_cc_band[1]]
        cc_note = "Motor hacmi bulunamadı; segment bazlı cc bandından tahmini MTV üretildi."
    else:
        cc_candidates = [engine_cc]
        cc_note = None

    candidates: List[Tuple[int, int, str]] = []
    for cc in cc_candidates:
        rows = [r for r in table if (r.get("cc_min", 0) <= cc <= r.get("cc_max", 999999))]
        if not rows:
            continue

        if is_value_based:
            if vehicle_value is None:
                vals = [int(r["tax"][age_key]) for r in rows if r.get("tax") and age_key in r["tax"]]
                if vals:
                    candidates.append((min(vals), max(vals), "I/value_unknown"))
            else:
                def in_value(r: Dict[str, Any]) -> bool:
                    vmin = r.get("value_min", 0) or 0
                    vmax = r.get("value_max", None)
                    if r.get("value_min_exclusive"):
                        okmin = vehicle_value > vmin
                    else:
                        okmin = vehicle_value >= vmin
                    okmax = True if vmax is None else (vehicle_value <= vmax)
                    return okmin and okmax

                rr = [r for r in rows if in_value(r)]
                if rr and rr[0].get("tax") and age_key in rr[0]["tax"]:
                    val = int(rr[0]["tax"][age_key])
                    candidates.append((val, val, "I/exact"))
                else:
                    vals = [int(r["tax"][age_key]) for r in rows if r.get("tax") and age_key in r["tax"]]
                    if vals:
                        candidates.append((min(vals), max(vals), "I/value_fallback"))
        else:
            r0 = rows[0]
            if r0.get("tax") and age_key in r0["tax"]:
                val = int(r0["tax"][age_key])
                candidates.append((val, val, "IA/exact"))

    if not candidates:
        return {"ok": False, "note": "MTV hesaplanamadı."}

    mtv_min = min(c[0] for c in candidates)
    mtv_max = max(c[1] for c in candidates)
    mtv_mid = int((mtv_min + mtv_max) / 2)

    notes = []
    if cc_note:
        notes.append(cc_note)
    if is_value_based and vehicle_value is None:
        notes.append("2018+ değer bazlı tarifede araç değeri bulunamadığı için MTV band olarak verildi.")

    return {
        "ok": True,
        "tax_year": ty,
        "age": age,
        "engine_cc_used": engine_cc,
        "vehicle_value_used_try": vehicle_value,
        "tariff_basis": sorted(set(c[2] for c in candidates)),
        "mtv_yearly_try_min": mtv_min,
        "mtv_yearly_try_max": mtv_max,
        "mtv_yearly_try_mid": mtv_mid,
        "mtv_installment_try_min": int(mtv_min / 2),
        "mtv_installment_try_max": int(mtv_max / 2),
        "mtv_installment_try_mid": int(mtv_mid / 2),
        "notes": notes,
    }


def estimate_vehicle_inspection(req: AnalyzeRequest) -> Dict[str, Any]:
    fc = FIXED_COSTS or {}
    vi = (fc.get("vehicle_inspection") or {})
    fees = (vi.get("fees_2026_estimated") or {}).get("car_light") or {}
    fee = fees.get("fee")
    if not isinstance(fee, (int, float)):
        return {"ok": False, "note": "Muayene ücreti datası bulunamadı."}

    annual = (vi.get("annualized_hint") or {}).get("car_light", {}).get("annual_equivalent_fee_2026_estimated")
    if not isinstance(annual, (int, float)):
        annual = float(fee) / 2.0

    return {
        "ok": True,
        "fee_try_2026_estimated": int(round(fee)),
        "annual_equivalent_try_2026_estimated": int(round(annual)),
        "notes": (vi.get("periodicity_notes") or [])[:2],
        "delay_fee_rule": vi.get("delay_fee_rule"),
    }


def estimate_traffic_insurance(req: AnalyzeRequest) -> Dict[str, Any]:
    caps = TRAFFIC_CAPS or {}
    cities = caps.get("cities") or {}
    city_code = str((req.context or {}).get("plate_city_code") or "34")
    if city_code not in cities:
        city_code = "34"

    step = (req.context or {}).get("traffic_step")
    try:
        step = str(int(step)) if step is not None else "4"
    except:
        step = "4"

    cap = (cities.get(city_code, {}).get("basamak", {}) or {}).get(step)
    if not isinstance(cap, int):
        return {"ok": False, "note": "Trafik tavan primi bulunamadı."}

    mult_min = 0.45 if step in ("7", "8") else 0.50
    mult_mid = 0.65 if step in ("7", "8") else 0.70
    tmin = int(cap * mult_min)
    tmid = int(cap * mult_mid)
    tmax = int(cap)

    return {
        "ok": True,
        "city_code": city_code,
        "city_name": cities.get(city_code, {}).get("name"),
        "traffic_step": step,
        "traffic_cap_try": cap,
        "traffic_est_try_min": tmin,
        "traffic_est_try_mid": tmid,
        "traffic_est_try_max": tmax,
        "note": "Trafik sigortası bandı ilgili il ve basamağa göre tahmini üretilmiştir; teklif sürücü geçmişine göre değişir.",
    }


def estimate_kasko(req: AnalyzeRequest, segment_code: str, age: Optional[int], mileage_km: int) -> Dict[str, Any]:
    listed_price = _parse_listed_price(req)
    if not listed_price or listed_price < 100_000:
        return {"ok": False, "note": "Kasko için araç değeri bulunamadı (fiyat yok veya düşük)."}

    base = {
        "B_HATCH": (0.020, 0.045),
        "C_SEDAN": (0.022, 0.050),
        "C_SUV": (0.025, 0.055),
        "D_SEDAN": (0.028, 0.065),
        "PREMIUM_D": (0.030, 0.080),
        "E_SEGMENT": (0.035, 0.095),
    }.get(segment_code, (0.022, 0.055))

    age_mult = 1.0
    if age is not None:
        if age >= 15:
            age_mult = 1.35
        elif age >= 10:
            age_mult = 1.22
        elif age >= 6:
            age_mult = 1.10

    km_mult = 1.0
    if mileage_km >= 250_000:
        km_mult = 1.25
    elif mileage_km >= 180_000:
        km_mult = 1.15
    elif mileage_km >= 120_000:
        km_mult = 1.08

    rmin = base[0] * age_mult * km_mult
    rmax = base[1] * age_mult * km_mult

    kmin = int(listed_price * rmin)
    kmax = int(listed_price * rmax)

    kmax = min(kmax, int(listed_price * 0.12))
    kmin = max(kmin, int(listed_price * 0.015))
    if kmin > kmax:
        kmin = int(kmax * 0.75)

    kmid = int((kmin + kmax) / 2)

    return {
        "ok": True,
        "kasko_try_min": kmin,
        "kasko_try_mid": kmid,
        "kasko_try_max": kmax,
        "note": "Kasko bandı araç değeri + segment + yaş/km ile tahmini üretilmiştir; teminatlar ve sürücü profiliyle değişir.",
    }


# =========================================================
# COST ESTIMATION (senin mantık)
# =========================================================
def estimate_costs(req: AnalyzeRequest) -> Dict[str, Any]:
    v = req.vehicle
    p = req.profile or Profile()

    segment_code = detect_segment(v.make, v.model)
    seg = SEGMENT_PROFILES.get(segment_code, SEGMENT_PROFILES["C_SEDAN"])

    listed_price = _parse_listed_price(req)

    current_year = int(os.getenv("CURRENT_YEAR", str(date.today().year)))
    age = None
    if v.year:
        age = max(0, current_year - v.year)
    mileage = int(v.mileage_km or 0)

    base_min, base_max = seg["maintenance_yearly_range"]

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
    fuel_comment = None
    if fuel == "diesel":
        fuel_mult_maint = 1.05
        fuel_comment = "Dizel motorda kısa mesafe/şehir içi ağırlığında bazı kalemler daha sık takip gerektirebilir."
    elif fuel == "lpg":
        fuel_mult_maint = 0.95
        fuel_comment = "LPG'li araçta montaj/ayar kalitesi ve düzenli kontrol, uzun vadeli performansı belirler."
    elif fuel in ("hybrid", "electric"):
        fuel_mult_maint = 0.9
        fuel_comment = "Hibrit/elektrikli araçlarda batarya sağlığı ve yetkili servis desteği önemli bir değişkendir."

    maint_min = int(base_min * age_mult * km_mult * usage_mult * fuel_mult_maint)
    maint_max = int(base_max * age_mult * km_mult * usage_mult * fuel_mult_maint)

    if listed_price and listed_price > 100_000:
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
            "B_HATCH": 45_000,
            "C_SEDAN": 55_000,
            "C_SUV": 75_000,
            "D_SEDAN": 85_000,
            "PREMIUM_D": 130_000,
            "E_SEGMENT": 160_000,
        }
        cap = caps.get(segment_code, 60_000)
        maint_max = min(maint_max, cap)

        if maint_min > maint_max:
            maint_min = int(maint_max * 0.7)

    mid_maint = int((maint_min + maint_max) / 2) if maint_max else maint_min
    routine_est = int(mid_maint * 0.65)
    unexpected_est = max(0, mid_maint - routine_est)  # “beklenmedik gider payı”

    km_year = max(0, int(p.yearly_km or 15_000))
    km_factor = km_year / 15_000.0
    km_factor = max(0.5, min(2.5, km_factor))

    fuel_base = {
        "B_HATCH": (18_000, 32_000),
        "C_SEDAN": (22_000, 40_000),
        "C_SUV": (26_000, 48_000),
        "D_SEDAN": (27_000, 52_000),
        "PREMIUM_D": (32_000, 65_000),
        "E_SEGMENT": (38_000, 80_000),
    }.get(segment_code, (22_000, 42_000))

    fuel_mult_cost = 1.0
    if fuel == "diesel":
        fuel_mult_cost = 0.9
    elif fuel == "lpg":
        fuel_mult_cost = 0.75
    elif fuel in ("hybrid", "electric"):
        fuel_mult_cost = 0.7

    fuel_min = int(fuel_base[0] * km_factor * fuel_mult_cost)
    fuel_max = int(fuel_base[1] * km_factor * fuel_mult_cost)
    fuel_mid = int((fuel_min + fuel_max) / 2) if fuel_max else fuel_min

    traffic = estimate_traffic_insurance(req)
    kasko = estimate_kasko(req, segment_code=segment_code, age=age, mileage_km=mileage)

    mtv = estimate_mtv(req)
    inspection = estimate_vehicle_inspection(req)

    return {
        "segment_code": segment_code,
        "segment_name": seg["name"],
        "age": age,
        "mileage_km": mileage,
        "listed_price_try": listed_price,
        "maintenance_yearly_try_min": maint_min,
        "maintenance_yearly_try_max": maint_max,
        "maintenance_routine_yearly_est": routine_est,
        "maintenance_unexpected_yearly_est": unexpected_est,
        "yearly_fuel_tr_min": fuel_min,
        "yearly_fuel_tr_max": fuel_max,
        "yearly_fuel_tr_mid": fuel_mid,
        "insurance_level": seg["insurance_level"],
        "insurance": {"traffic": traffic, "kasko": kasko},
        "fuel_comment": fuel_comment,
        "segment_notes": seg.get("notes", []),
        "consumption_l_per_100km_band": {
            "B_HATCH": (5.5, 7.5),
            "C_SEDAN": (6.5, 9.0),
            "C_SUV": (7.5, 11.0),
            "D_SEDAN": (7.5, 11.0),
            "PREMIUM_D": (8.0, 12.5),
            "E_SEGMENT": (9.0, 15.0),
        }.get(segment_code, (6.5, 9.5)),
        "taxes": {"mtv": mtv},
        "fixed_costs": {"inspection": inspection},
    }


def build_enriched_context(req: AnalyzeRequest) -> Dict[str, Any]:
    v = req.vehicle
    p = req.profile or Profile()

    seg_info = estimate_costs(req)
    segment_code = seg_info["segment_code"]

    anchors = find_anchor_matches(v.make, v.model, segment_code, limit=3)
    info_q = evaluate_info_quality(req)

    yearly_km_band = "orta"
    if p.yearly_km <= 8000:
        yearly_km_band = "düşük"
    elif p.yearly_km >= 30000:
        yearly_km_band = "yüksek"

    prof = resolve_vehicle_profile(req, segment_code)
    vp = prof["profile"] or {}
    guess_tags = _guess_tags(req)

    # checklist
    checklist = vp.get("inspection_checklist") or []
    if not checklist:
        cl = _load_json(_dp("inspection_checklists_by_segment_v1.json"), {})
        checklist = cl.get(segment_code, [])

    return {
        "segment": {
            "code": segment_code,
            "name": seg_info["segment_name"],
            "notes": seg_info.get("segment_notes", []),
        },
        "market": {
            "indices": (vp.get("indices") or {}),
            "profile_match": prof.get("match"),
            "matched_key": prof.get("matched_key"),
        },
        "costs": {
            "listed_price_try": seg_info["listed_price_try"],
            "maintenance_yearly_try_min": seg_info["maintenance_yearly_try_min"],
            "maintenance_yearly_try_max": seg_info["maintenance_yearly_try_max"],
            "maintenance_routine_yearly_est": seg_info["maintenance_routine_yearly_est"],
            "maintenance_unexpected_yearly_est": seg_info["maintenance_unexpected_yearly_est"],
            "yearly_fuel_tr_min": seg_info["yearly_fuel_tr_min"],
            "yearly_fuel_tr_max": seg_info["yearly_fuel_tr_max"],
            "yearly_fuel_tr_mid": seg_info["yearly_fuel_tr_mid"],
            "consumption_l_per_100km_band": seg_info["consumption_l_per_100km_band"],
            "fuel_comment": seg_info.get("fuel_comment"),
        },
        "insurance": seg_info.get("insurance", {}),
        "taxes": seg_info.get("taxes", {}),
        "fixed_costs": seg_info.get("fixed_costs", {}),
        "profile": {
            "yearly_km": p.yearly_km,
            "yearly_km_band": yearly_km_band,
            "usage": p.usage,
            "fuel_preference": p.fuel_preference,
            "family": getattr(p, "family", "couple"),
            "budget_sensitivity": getattr(p, "budget_sensitivity", "medium"),
            "priority": getattr(p, "priority", "balance"),
        },
        "info_quality": info_q,
        "anchors_used": [
            {"key": a.get("key"), "segment": a.get("segment"), "tags": a.get("tags", []), "parts": a.get("parts"), "resale": a.get("resale")}
            for a in anchors
        ],
        "inspection_checklist": (checklist or [])[:10],
        "tags_detected": guess_tags,
    }


# =========================================================
# SCORING (YENİ) - 10 üzerinden + 100 üzerinden
# =========================================================
def _map_1_5_to_10(v: float) -> float:
    # 1..5 => 2..10 gibi lineer
    v = _clamp(float(v), 1.0, 5.0)
    return 2.0 + (v - 1.0) * 2.0  # 1->2, 5->10

def _to_100(x10: float) -> int:
    return int(round(_clamp(x10, 0.0, 10.0) * 10.0))

def compute_scores(req: AnalyzeRequest, enriched: Dict[str, Any]) -> Dict[str, Any]:
    v = req.vehicle
    p = req.profile
    seg_code = enriched["segment"]["code"]
    idx = (enriched.get("market", {}) or {}).get("indices", {}) or {}
    costs = enriched.get("costs", {}) or {}

    age = None
    current_year = int(os.getenv("CURRENT_YEAR", str(date.today().year)))
    if v.year:
        age = max(0, current_year - v.year)
    km = int(v.mileage_km or 0)

    fuel = (v.fuel or p.fuel_preference or "unknown").lower().strip()
    usage = (p.usage or "mixed").lower().strip()
    family = (getattr(p, "family", "couple") or "couple").lower().strip()
    budget_sens = (getattr(p, "budget_sensitivity", "medium") or "medium").lower().strip()
    priority = (getattr(p, "priority", "balance") or "balance").lower().strip()

    # ---- Mekanik (10)
    mech = 8.0
    if age is not None:
        if age >= 15: mech -= 2.0
        elif age >= 10: mech -= 1.2
        elif age >= 6:  mech -= 0.6
    if km >= 250_000: mech -= 2.0
    elif km >= 180_000: mech -= 1.2
    elif km >= 120_000: mech -= 0.6

    # yakıt + kullanım hassasiyeti
    if fuel == "diesel" and usage == "city":
        mech -= 0.6
    if fuel == "lpg":
        mech -= 0.3  # montaj/ayar değişkenliği
    if fuel in ("hybrid", "electric"):
        mech -= 0.2

    if seg_code in ("PREMIUM_D", "E_SEGMENT"):
        mech -= 0.4  # bakım hassasiyeti
    mech = _clamp(mech, 2.5, 9.5)

    # ---- Elektronik/Donanım (10)
    elec = 7.6
    if seg_code in ("PREMIUM_D", "E_SEGMENT"):
        elec -= 0.7
    if age is not None:
        if age >= 15: elec -= 1.2
        elif age >= 10: elec -= 0.8
        elif age >= 6:  elec -= 0.4
    elec = _clamp(elec, 2.5, 9.5)

    # ---- Maliyet Dengesi (10) (segment + endeks + band)
    parts_cost = float(idx.get("parts_cost_index_1_5", 3))
    base_cost = 8.0 - (parts_cost - 1.0) * 1.2  # parça maliyet endeksi yükseldikçe düşer
    if seg_code in ("PREMIUM_D", "E_SEGMENT"):
        base_cost -= 0.8
    if budget_sens == "high":
        base_cost -= 0.6
    # bakım bandı çok genişse biraz düşür
    mmin = int(costs.get("maintenance_yearly_try_min") or 0)
    mmax = int(costs.get("maintenance_yearly_try_max") or 0)
    if mmax > 0 and mmin > 0:
        spread = (mmax - mmin) / max(1, mmax)
        if spread >= 0.45:
            base_cost -= 0.5
    cost_balance = _clamp(base_cost, 2.5, 9.5)

    # ---- Likidite (10)
    resale = float(idx.get("resale_liquidity_score_1_5", 3))
    liquidity = _clamp(_map_1_5_to_10(resale), 2.5, 9.5)

    # ---- Kişiye Uygunluk (10) (tamamen "uyum", yargı yok)
    fit = 7.5
    yearly_km = int(p.yearly_km or 15000)

    # yakıt + yıllık km uyumu
    if yearly_km <= 8000 and fuel == "diesel":
        fit -= 0.8
    if yearly_km >= 30000 and fuel == "gasoline":
        fit -= 0.7
    if yearly_km >= 25000 and fuel == "lpg":
        fit += 0.4
    if fuel in ("hybrid", "electric") and usage in ("city", "mixed"):
        fit += 0.3

    # kullanım tipi etkisi
    if usage == "city":
        if seg_code in ("C_SUV", "D_SEDAN", "PREMIUM_D", "E_SEGMENT"):
            fit -= 0.3  # park/yoğun şehir
    if usage == "highway":
        fit += 0.2

    # aile durumu
    if family == "kids":
        if seg_code in ("B_HATCH",):
            fit -= 0.7
        if seg_code in ("C_SEDAN", "C_SUV", "D_SEDAN"):
            fit += 0.3

    # öncelik
    if priority == "comfort" and seg_code in ("D_SEDAN", "PREMIUM_D", "E_SEGMENT"):
        fit += 0.3
    if priority == "performance" and fuel in ("hybrid", "electric"):
        fit += 0.2

    fit = _clamp(fit, 2.5, 9.8)

    # ---- Genel skor (10) - ağırlıklı
    overall = (
        mech * 0.30 +
        elec * 0.20 +
        cost_balance * 0.20 +
        liquidity * 0.15 +
        fit * 0.15
    )
    overall = _clamp(overall, 2.5, 9.8)

    return {
        "overall_10": round(overall, 1),
        "mechanical_10": round(mech, 1),
        "electronics_10": round(elec, 1),
        "cost_balance_10": round(cost_balance, 1),
        "liquidity_10": round(liquidity, 1),
        "personal_fit_10": round(fit, 1),

        "overall_100": _to_100(overall),
        "mechanical_100": _to_100(mech),
        "electronics_100": _to_100(elec),
        "cost_balance_100": _to_100(cost_balance),
        "liquidity_100": _to_100(liquidity),
        "personal_fit_100": _to_100(fit),
    }


def build_fit_explanation(req: AnalyzeRequest, enriched: Dict[str, Any], fit10: float) -> str:
    p = req.profile
    v = req.vehicle
    seg = enriched["segment"]["name"]
    fuel = (v.fuel or p.fuel_preference or "unknown")
    usage = p.usage
    yearly_km = p.yearly_km

    lines = []
    lines.append(f"- Profil: **{yearly_km} km/yıl**, kullanım: **{usage}**, tercih: **{fuel}**")
    lines.append(f"- Segment: **{seg}**")
    # nötr, yargısız açıklama
    lines.append("Bu skor; kullanım alışkanlığı, yıllık km bandı, yakıt türü ve segment pratikliği birlikte düşünülerek üretilmiş göreceli bir uyum göstergesidir.")
    return "\n".join(lines)


# =========================================================
# PREMIUM OUTPUT (YENİ: skor + kişiye uygunluk)
# =========================================================
def build_premium_markdown(req: AnalyzeRequest, enriched: Dict[str, Any], scores: Dict[str, Any]) -> str:
    v = req.vehicle
    p = req.profile
    title = f"{v.year or ''} {v.make} {v.model}".strip() or "Premium Analiz"

    iq = enriched.get("info_quality", {})
    costs = enriched.get("costs", {})
    ins = enriched.get("insurance", {})
    taxes = enriched.get("taxes", {})
    fixed = enriched.get("fixed_costs", {})
    idx = (enriched.get("market", {}) or {}).get("indices", {}) or {}

    maint_min = costs.get("maintenance_yearly_try_min")
    maint_max = costs.get("maintenance_yearly_try_max")
    fuel_min = costs.get("yearly_fuel_tr_min")
    fuel_max = costs.get("yearly_fuel_tr_max")
    fuel_mid = costs.get("yearly_fuel_tr_mid")

    mtv = (taxes.get("mtv") or {})
    insp = (fixed.get("inspection") or {})
    traffic = (ins.get("traffic") or {})
    kasko = (ins.get("kasko") or {})

    parts_av = idx.get("parts_availability_score_1_5", 3)
    parts_cost = idx.get("parts_cost_index_1_5", 3)
    service_net = idx.get("service_network_index_1_5", 3)
    resale_liq = idx.get("resale_liquidity_score_1_5", 3)

    checklist = enriched.get("inspection_checklist", []) or []
    checklist = checklist[:6]

    lines = []
    lines.append(f"## {title}")
    if enriched.get("market", {}).get("profile_match") == "segment_estimate":
        lines.append("**Not:** Model profili bulunamadı; aynı segment emsallerinden tahmini endeksler kullanıldı.")
    lines.append(f"**Bilgi seviyesi:** {iq.get('level','-')} (eksikler: {', '.join(iq.get('missing_fields', [])[:6]) or 'yok'})")
    lines.append("")

    # Skorlar
    lines.append("### 1) Oto Analiz Skoru")
    lines.append(f"**{scores['overall_10']} / 10**  _(100’lük karşılığı: {scores['overall_100']})_")
    lines.append("")
    lines.append("### 2) Alt Skorlar")
    lines.append(f"- Mekanik: **{scores['mechanical_10']} / 10**")
    lines.append(f"- Elektronik/Donanım: **{scores['electronics_10']} / 10**")
    lines.append(f"- Maliyet Dengesi: **{scores['cost_balance_10']} / 10**")
    lines.append(f"- 2. El Likidite: **{scores['liquidity_10']} / 10**")
    lines.append("")

    # Kişiye uygunluk
    lines.append("### 3) Kişiye Uygunluk")
    lines.append(f"**{scores['personal_fit_10']} / 10**  _(100’lük karşılığı: {scores['personal_fit_100']})_")
    lines.append(build_fit_explanation(req, enriched, float(scores["personal_fit_10"])))
    lines.append("")

    # Maliyet
    lines.append("### 4) Yıllık maliyet özeti (tahmini band)")
    if maint_min and maint_max:
        lines.append(f"- Bakım: **{_fmt_try(maint_min)} – {_fmt_try(maint_max)} TL/yıl**")
        lines.append(f"  - Periyodik pay: ~**{_fmt_try(costs.get('maintenance_routine_yearly_est'))} TL/yıl**")
        lines.append(f"  - Beklenmedik gider payı: ~**{_fmt_try(costs.get('maintenance_unexpected_yearly_est'))} TL/yıl**")
    if fuel_min and fuel_max:
        lines.append(f"- Yakıt/enerji: **{_fmt_try(fuel_min)} – {_fmt_try(fuel_max)} TL/yıl** (orta: {_fmt_try(fuel_mid)} TL)")
    if mtv.get("ok"):
        lines.append(f"- MTV: **{_fmt_try(mtv.get('mtv_yearly_try_min'))} – {_fmt_try(mtv.get('mtv_yearly_try_max'))} TL/yıl**")
    if insp.get("ok"):
        lines.append(f"- Muayene yıllık ortalama: **~{_fmt_try(insp.get('annual_equivalent_try_2026_estimated'))} TL/yıl**")

    if traffic.get("ok"):
        lines.append(f"- Trafik sigortası: **{_fmt_try(traffic.get('traffic_est_try_min'))} – {_fmt_try(traffic.get('traffic_est_try_max'))} TL/yıl**")
    if kasko.get("ok"):
        lines.append(f"- Kasko: **{_fmt_try(kasko.get('kasko_try_min'))} – {_fmt_try(kasko.get('kasko_try_max'))} TL/yıl**")
    fc = costs.get("fuel_comment")
    if fc:
        lines.append(f"- Not: {fc}")
    lines.append("")

    # Parça / servis / 2.el
    lines.append("### 5) Parça, servis ve ikinci el endeksleri")
    lines.append(f"- Parça bulunabilirliği: **{parts_av}/5**")
    lines.append(f"- Parça maliyet endeksi: **{parts_cost}/5**")
    lines.append(f"- Servis ağı: **{service_net}/5**")
    lines.append(f"- 2. el likidite: **{resale_liq}/5**")
    lines.append("")

    # checklist
    lines.append("### 6) Kontrol listesi (kısa)")
    if checklist:
        for c in checklist:
            lines.append(f"- {c}")
    else:
        lines.append("- Ekspertiz + tramer + OBD taraması ile temel kontrolleri tamamla.")
    lines.append("")

    lines.append("### 7) Notlar")
    lines.append("Bu rapor; segment emsalleri, yaş/km ve kullanım profiline göre **band** üretir. İlan detayları arttıkça bandlar daralır ve skorlar daha isabetli hale gelir.")

    text = "\n".join(lines)

    # ekstra güvenlik: yasaklı kelimeler sızmasın
    if _contains_banned(text):
        # son çare: yasaklı kelimeleri maskeler
        for w in BANNED_WORDS:
            text = re.sub(w, "—", text, flags=re.IGNORECASE)

    return text


# =========================================================
# OPTIONAL LLM REWRITE (sadece metin, yasaklara uyacak)
# =========================================================
def call_llm_json(model_name: str, system_prompt: str, user_content: str) -> Optional[Dict[str, Any]]:
    if client is None:
        return None
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
        return json.loads(content) if isinstance(content, str) else content
    except Exception as e:
        print("LLM error:", e)
        return None


SYSTEM_PROMPT_PREMIUM_REWRITE = """
Türkçe yaz.
Aşağıdaki metinleri daha doğal, daha premium hale getir:
- summary.short_comment
- summary.pros
- summary.cons
- summary.notes (kısa)
Kurallar:
- 'alınır/alınmaz/tehlikeli/riskli/sakın' gibi ifadeler KULLANMA.
- Kesin yargı verme; yönlendirme yapma.
- Rakam uydurma, rakam değiştirme.
ÇIKTI sadece JSON:
{
  "summary": {
    "short_comment": "",
    "pros": [],
    "cons": [],
    "notes": ""
  }
}
""".strip()


# =========================================================
# PREMIUM IMPLEMENTATION
# =========================================================
def premium_analyze_impl(req: AnalyzeRequest) -> Dict[str, Any]:
    enriched = build_enriched_context(req)
    scores = compute_scores(req, enriched)
    md = build_premium_markdown(req, enriched, scores)

    # summary alanını LLM ile iyileştir (opsiyonel)
    summary = {
        "short_comment": "Skorlar, maliyet bandı ve endekslerle özet hazırlandı.",
        "pros": [
            "Skorlar: genel + alt başlıklar + kişiye uygunluk birlikte verildi.",
            "Yıllık maliyet bandı bakım/yakıt/sigorta/vergiler şeklinde ayrıştırıldı.",
            "Parça/servis ve 2. el endeksleri eklendi."
        ],
        "cons": [
            "İlan detayları kısa ise skorlar daha geniş belirsizlikle hesaplanır.",
            "Bakım geçmişi ve tramer netleşirse skorlar daha isabetli olur."
        ],
        "notes": "Skorlar görecelidir; daha fazla ilan detayı bandları daraltır."
    }

    # LLM varsa sadece summary'yi parlat
    llm = call_llm_json(
        OPENAI_MODEL_PREMIUM,
        SYSTEM_PROMPT_PREMIUM_REWRITE,
        json.dumps({
            "vehicle": req.vehicle.dict(),
            "profile": req.profile.dict(),
            "info_quality": enriched.get("info_quality"),
            "base_summary": summary
        }, ensure_ascii=False)
    )
    if isinstance(llm, dict) and isinstance(llm.get("summary"), dict):
        cand = llm["summary"]
        # güvenlik: yasaklı kelime kontrolü
        blob = json.dumps(cand, ensure_ascii=False).lower()
        if not _contains_banned(blob):
            summary = {
                "short_comment": cand.get("short_comment", summary["short_comment"]),
                "pros": cand.get("pros", summary["pros"]),
                "cons": cand.get("cons", summary["cons"]),
                "notes": cand.get("notes", summary["notes"]),
            }

    return {
        "scores": scores,
        "summary": summary,
        "preview": {
            "title": f"{req.vehicle.year or ''} {req.vehicle.make} {req.vehicle.model}".strip() or "Premium Analiz",
            "bullets": [
                f"Oto Analiz Skoru: {scores['overall_10']}/10",
                f"Kişiye Uygunluk: {scores['personal_fit_10']}/10",
                "Maliyet bandı (bakım + yakıt + sigorta + vergiler)",
                "Parça/servis + 2. el endeksleri"
            ]
        },
        "data": {
            "enriched": enriched  # istersen Flutter kullanır; kullanmazsan da sorun değil
        },
        "result": md
    }


# =========================================================
# NORMAL / MANUAL / COMPARE / OTOBOT (senin gibi, sadece yasak filtresi ek)
# =========================================================
SYSTEM_PROMPT_NORMAL = """
Türkçe yaz ve SADECE JSON döndür.
Kurallar:
- 'alınır/alınmaz/tehlikeli/riskli/sakın' yazma.
- Kesin yargı verme; yönlendirme yapma.
Şablon:
{
  "scores": {"overall_10":0,"mechanical_10":0,"electronics_10":0,"cost_balance_10":0,"liquidity_10":0,"personal_fit_10":0},
  "summary": {"short_comment":"","pros":[],"cons":[],"notes":""},
  "result":""
}
""".strip()


SYSTEM_PROMPT_COMPARE = """
Türkçe yaz ve SADECE JSON döndür.
Kurallar:
- 'alınır/alınmaz/tehlikeli/riskli/sakın' yazma.
Şablon:
{
  "comparison":"left",
  "notes":"",
  "left_points":[],
  "right_points":[]
}
""".strip()


SYSTEM_PROMPT_OTOBOT = """
Türkçe yaz ve SADECE JSON döndür.
Kurallar:
- 'alınır/alınmaz/tehlikeli/riskli/sakın' yazma.
Şablon:
{"answer":"","suggested_segments":[],"example_models":[]}
""".strip()


def fallback_normal(req: AnalyzeRequest) -> Dict[str, Any]:
    enriched = build_enriched_context(req)
    scores = compute_scores(req, enriched)
    md = build_premium_markdown(req, enriched, scores)  # normalde de aynı format dönebilir
    return {
        "scores": {
            "overall_10": scores["overall_10"],
            "mechanical_10": scores["mechanical_10"],
            "electronics_10": scores["electronics_10"],
            "cost_balance_10": scores["cost_balance_10"],
            "liquidity_10": scores["liquidity_10"],
            "personal_fit_10": scores["personal_fit_10"],
        },
        "summary": {
            "short_comment": "Sınırlı bilgiyle skorlar ve maliyet bandı üretildi.",
            "pros": ["Skorlar ve maliyet bandı hazırlandı."],
            "cons": ["İlan detayları arttıkça sonuçlar daha isabetli olur."],
            "notes": "Detaylı doğrulama için ekspertiz + tramer + bakım kayıtları birlikte değerlendirilmelidir."
        },
        "result": md
    }


def safe_llm_call(model: str, system_prompt: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    out = call_llm_json(model, system_prompt, json.dumps(payload, ensure_ascii=False))
    if not isinstance(out, dict):
        return None
    # yasaklı kelime kontrolü (string alanlarda)
    blob = json.dumps(out, ensure_ascii=False)
    if _contains_banned(blob):
        return None
    return out


# =========================================================
# HEALTH
# =========================================================
@app.get("/")
async def root() -> Dict[str, Any]:
    return {"ok": True, "message": "Oto Analiz backend çalışıyor."}


# =========================================================
# ENDPOINTS
# =========================================================
@app.post("/analyze")
async def analyze(req: AnalyzeRequest) -> Dict[str, Any]:
    enriched = build_enriched_context(req)
    scores = compute_scores(req, enriched)

    payload = {
        "vehicle": req.vehicle.dict(),
        "profile": req.profile.dict(),
        "scores_hint": scores,
        "info_quality": enriched.get("info_quality"),
        "notes": "Kesin yargı yok; sadece skor + kısa açıklama."
    }

    out = safe_llm_call(OPENAI_MODEL_NORMAL, SYSTEM_PROMPT_NORMAL, payload)
    if out:
        return out

    return fallback_normal(req)


@app.post("/premium_analyze")
async def premium_analyze(req: AnalyzeRequest) -> Dict[str, Any]:
    return premium_analyze_impl(req)


@app.post("/manual_analyze")
@app.post("/manual")
async def manual_analyze(req: AnalyzeRequest) -> Dict[str, Any]:
    # manueli şu an normal gibi ele alıyoruz (istersen ayrı prompt yazarız)
    return await analyze(req)


@app.post("/compare_analyze")
async def compare_analyze(req: CompareRequest) -> Dict[str, Any]:
    payload = {
        "left": {"vehicle": req.left.vehicle.dict(), "ad_description": req.left.ad_description or ""},
        "right": {"vehicle": req.right.vehicle.dict(), "ad_description": req.right.ad_description or ""},
        "profile": req.profile.dict() if req.profile else None,
        "notes": "Kesin yargı yok; sadece öne çıkan farklılıklar."
    }

    out = safe_llm_call(OPENAI_MODEL_COMPARE, SYSTEM_PROMPT_COMPARE, payload)
    if out:
        return out

    return {
        "comparison": "left",
        "notes": "İki tarafta da ilan detayları ve bakım geçmişi netleşince karşılaştırma daha doğru olur.",
        "left_points": ["Skor/masraf bandını netleştirmek için ekspertiz + tramer birlikte düşünülmeli."],
        "right_points": ["İlan detayları arttıkça karşılaştırma daha isabetli olur."]
    }


@app.post("/otobot")
async def otobot(req: OtoBotRequest) -> Dict[str, Any]:
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Soru boş olamaz. 'question' alanına bir metin gönder.")

    out = safe_llm_call(OPENAI_MODEL_OTOBOT, SYSTEM_PROMPT_OTOBOT, {"question": question, "history": req.history or []})
    if out:
        return out

    return {
        "answer": "Bütçe, yıllık km ve kullanım tipine göre segment seçmek en mantıklı başlangıçtır. İlan detaylarıyla skoru netleştirebilirsin.",
        "suggested_segments": ["C-sedan", "C-SUV", "B-hatch"],
        "example_models": ["Toyota Corolla", "Renault Megane", "Honda Civic", "Hyundai Tucson"]
    }
