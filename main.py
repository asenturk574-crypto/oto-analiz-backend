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
    yearly_km: int = Field(15000, ge=0, le=100000)
    usage: str = "mixed"              # city / mixed / highway
    fuel_preference: str = "gasoline" # gasoline / diesel / lpg / hybrid / electric

    # --- kişiye uygunluk için opsiyonel (Flutter göndermese de default var) ---
    family: str = "unknown"           # single / couple / kids / unknown
    budget_sensitivity: str = "medium"  # low / medium / high
    priority: str = "balance"         # comfort / performance / economy / balance

    class Config:
        extra = "allow"


class Vehicle(BaseModel):
    make: str = ""
    model: str = ""
    year: Optional[int] = Field(None, ge=1980, le=2035)
    mileage_km: Optional[int] = Field(None, ge=0)
    fuel: Optional[str] = None        # gasoline / diesel / lpg / hybrid / electric

    # opsiyonel
    transmission: Optional[str] = None  # manual / auto / dsg / cvt / unknown
    drive: Optional[str] = None         # fwd / rwd / awd / 4x4 / unknown
    engine_cc: Optional[int] = None

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


def _clamp(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, n))


def _clampf(n: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, n))


def _to10(score100: int) -> float:
    return round(score100 / 10.0, 1)


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


def _sanitize_no_verdict(text: str) -> str:
    """
    'alınır/alınmaz/tehlikeli/riskli/sakın' gibi kelimeler sızarsa yumuşat.
    """
    if not text:
        return text

    repl = {
        "alınır": "uygunluk skoru",
        "alınmaz": "uygunluk skoru",
        "sakın": "dikkat",
        "tehlikeli": "dikkat noktası",
        "riskli": "dikkat gerektiren",
        "kesinlikle": "genellikle",
    }

    t = text
    for k, v in repl.items():
        t = re.sub(rf"\b{k}\b", v, t, flags=re.IGNORECASE)

    return t


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
            stats[seg][k] = (sum(arr) / max(1, len(arr))) if arr else 3.0
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
            "Şehir içi kullanımda pratik; masraflar genelde daha kontrol edilebilir olur.",
        ],
    },
    "C_SEDAN": {
        "name": "C segment (aile sedan/hatchback)",
        "maintenance_yearly_range": (15000, 32000),
        "insurance_level": "orta",
        "notes": [
            "Türkiye’de en likit segmentlerden; temiz örneklerin alıcısı genelde vardır.",
        ],
    },
    "C_SUV": {
        "name": "C segment SUV",
        "maintenance_yearly_range": (18000, 38000),
        "insurance_level": "orta-yüksek",
        "notes": [
            "Lastik/fren/alt takım maliyeti kompakt araçlara göre artabilir.",
        ],
    },
    "D_SEDAN": {
        "name": "D segment (konfor sedan)",
        "maintenance_yearly_range": (22000, 50000),
        "insurance_level": "orta-yüksek",
        "notes": [
            "Konfor artarken parça/işçilik maliyetleri C segmente göre yükselir.",
        ],
    },
    "PREMIUM_D": {
        "name": "Premium sınıf",
        "maintenance_yearly_range": (32000, 75000),
        "insurance_level": "yüksek",
        "notes": [
            "Premium sınıfta işçilik/parça maliyeti ve elektronik donanım kalemleri daha pahalıdır.",
        ],
    },
    "E_SEGMENT": {
        "name": "E segment / üst sınıf",
        "maintenance_yearly_range": (45000, 120000),
        "insurance_level": "çok yüksek",
        "notes": [
            "Büyük gövde ve yüksek donanım: masraf kalemleri belirgin şekilde yükselir.",
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
    if any(k in s for k in ["qashqai", "tucson", "sportage", "kuga", "3008", "duster", "qq"]):
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

        if any(m in target for m in ["bmw", "mercedes", "audi", "volvo"]) and any(m in key for m in ["bmw", "mercedes", "audi", "volvo"]):
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
        "disclaimer": "Bu araç için doğrudan model profili bulunamadı; aynı segmentteki emsallerden tahmini endeksler üretildi.",
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
# TAXES + FIXED COSTS
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

    engine_cc = (req.context or {}).get("engine_cc") or v.engine_cc
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


# =========================================================
# COST + INSURANCE ESTIMATION (band)
# =========================================================
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
        "note": "Trafik sigortası bandı; il ve basamak tavanına göre tahmini verilir. Gerçek teklif sürücü/hasar geçmişine göre değişir.",
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
        "note": "Kasko bandı; araç değeri + segment + yaş/km ile tahmini verilir. Şehir/teminat/hasarsızlık indirimiyle ciddi değişebilir.",
    }


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
    fuel_comment = "orta"

    if fuel == "diesel":
        fuel_mult_maint = 1.05
        if p.usage == "city" and mileage >= 120_000:
            fuel_comment = "orta-yüksek (DPF/EGR/enjektör takibi şehir içiyle daha kritikleşebilir)"
    elif fuel == "lpg":
        fuel_mult_maint = 0.95
        fuel_comment = "orta (montaj/ayar kalitesi önemli; subap takibi gerekir)"
    elif fuel in ("hybrid", "electric"):
        fuel_mult_maint = 0.9
        fuel_comment = "düşük-orta (batarya sağlığına bağlı)"

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
    reserve_est = max(0, mid_maint - routine_est)

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

    baseline = "orta"
    notes: List[str] = []

    if age is not None and age >= 15:
        baseline = "yüksek"
        notes.append("Araç yaşı yüksek; masraf belirsizliği artabilir.")
    elif age is not None and age >= 10:
        baseline = "orta-yüksek"
        notes.append("Yaş nedeniyle büyük bakım ihtimali dönemsel artabilir.")

    if mileage >= 250_000:
        baseline = "yüksek"
        notes.append("Km çok yüksek; mekanik yıpranma ihtimali artabilir.")
    elif mileage >= 180_000 and baseline != "yüksek":
        baseline = "orta-yüksek"
        notes.append("Km yüksek; yürüyen aksam ve mekanik kalemler daha yakından izlenmeli.")

    if segment_code in ("PREMIUM_D", "E_SEGMENT") and ((age and age > 10) or mileage > 180_000):
        notes.append("Premium/üst sınıfta büyük bakım kalemleri daha pahalı olabilir.")

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
        "maintenance_risk_reserve_yearly_est": reserve_est,
        "yearly_fuel_tr_min": fuel_min,
        "yearly_fuel_tr_max": fuel_max,
        "yearly_fuel_tr_mid": fuel_mid,
        "insurance_level": seg["insurance_level"],
        "insurance": {"traffic": traffic, "kasko": kasko},
        "fuel_comment": fuel_comment,
        "baseline_level": baseline,
        "notes": notes,
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


# =========================================================
# ENRICHED CONTEXT
# =========================================================
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
    base_risks = (vp.get("risk_patterns") or [])
    base_watch = (vp.get("big_maintenance_watchlist") or [])

    if not base_risks:
        tag_map = _load_json(_dp("risk_patterns_by_tag_v1.json"), {})
        for t in guess_tags:
            base_risks.extend(tag_map.get(t, []))
    if not base_watch:
        watch_map = _load_json(_dp("big_maintenance_watchlist_by_tag_v1.json"), {})
        for t in guess_tags:
            base_watch.extend(watch_map.get(t, []))

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
            "indices": vp.get("indices") or {},
            "profile_match": prof.get("match"),
            "matched_key": prof.get("matched_key"),
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
            "consumption_l_per_100km_band": seg_info["consumption_l_per_100km_band"],
        },
        "insurance": seg_info.get("insurance", {}),
        "taxes": seg_info.get("taxes", {}),
        "fixed_costs": seg_info.get("fixed_costs", {}),
        "risk": {
            "fuel_comment": seg_info["fuel_comment"],
            "baseline_level": seg_info["baseline_level"],
            "notes": seg_info["notes"],
            "age": seg_info["age"],
            "mileage_km": seg_info["mileage_km"],
            "risk_patterns": base_risks[:12],
            "maintenance_watchlist": base_watch[:12],
        },
        "profile": {
            "yearly_km": p.yearly_km,
            "yearly_km_band": yearly_km_band,
            "usage": p.usage,
            "fuel_preference": p.fuel_preference,
            "family": getattr(p, "family", "unknown"),
            "budget_sensitivity": getattr(p, "budget_sensitivity", "medium"),
            "priority": getattr(p, "priority", "balance"),
        },
        "info_quality": info_q,
        "anchors_used": [
            {"key": a.get("key"), "segment": a.get("segment"), "tags": a.get("tags", [])}
            for a in anchors
        ],
        "inspection_checklist": checklist[:10],
        "tags_detected": guess_tags,
    }


# =========================================================
# SCORING (10'luk + 100'lük)
# =========================================================
def _compute_overall_score(req: AnalyzeRequest, enriched: Dict[str, Any]) -> Dict[str, Any]:
    risk = enriched.get("risk", {}) or {}
    prof = enriched.get("profile", {}) or {}
    idx = (enriched.get("market", {}) or {}).get("indices", {}) or {}
    seg_code = (enriched.get("segment", {}) or {}).get("code", "C_SEDAN")
    iq = enriched.get("info_quality", {}) or {}

    age = risk.get("age")
    km = int(risk.get("mileage_km") or 0)
    baseline = str(risk.get("baseline_level") or "orta")

    parts_av = int(idx.get("parts_availability_score_1_5", 3) or 3)
    parts_cost = int(idx.get("parts_cost_index_1_5", 3) or 3)
    service_net = int(idx.get("service_network_index_1_5", 3) or 3)
    resale_liq = int(idx.get("resale_liquidity_score_1_5", 3) or 3)

    score = 78

    # info quality
    if iq.get("level") == "orta":
        score -= 3
    elif iq.get("level") == "düşük":
        score -= 8

    # baseline level
    if baseline == "düşük":
        score += 3
    elif baseline == "orta-yüksek":
        score -= 6
    elif baseline == "yüksek":
        score -= 12

    # age penalty
    if isinstance(age, int):
        if age >= 15:
            score -= 16
        elif age >= 10:
            score -= 10
        elif age >= 6:
            score -= 6
        elif age >= 3:
            score -= 3

    # km penalty
    if km >= 250_000:
        score -= 14
    elif km >= 180_000:
        score -= 10
    elif km >= 120_000:
        score -= 6
    elif km >= 80_000:
        score -= 3

    # market indices effect
    score += (resale_liq - 3) * 2
    score += (service_net - 3) * 1
    score += (parts_av - 3) * 1
    score -= (parts_cost - 3) * 5

    # premium segments slightly stricter
    if seg_code in ("PREMIUM_D", "E_SEGMENT"):
        score -= 4

    score = _clamp(score, 35, 90)

    # Subscores
    mechanical = _clamp(score + 5 - (0 if km < 120_000 else 4) - (0 if (age is None or age < 10) else 3), 30, 92)
    electronics = _clamp(score + 6 - (0 if (age is None or age < 10) else 5), 28, 90)

    # cost balance: heavy penalty by parts_cost + premium
    cost_balance = 75 - (parts_cost - 1) * 12
    if seg_code in ("PREMIUM_D", "E_SEGMENT"):
        cost_balance -= 8
    if isinstance(age, int) and age >= 10:
        cost_balance -= 6
    if km >= 180_000:
        cost_balance -= 6
    cost_balance = _clamp(cost_balance, 18, 85)

    liquidity = _clamp(int(resale_liq * 20), 20, 95)

    return {
        "overall_100": int(score),
        "mechanical_100": int(mechanical),
        "electronics_100": int(electronics),
        "cost_balance_100": int(cost_balance),
        "liquidity_100": int(liquidity),
        "indices_used": {
            "parts_av": parts_av,
            "parts_cost": parts_cost,
            "service_net": service_net,
            "resale_liq": resale_liq,
        }
    }


def _compute_personal_fit(req: AnalyzeRequest, enriched: Dict[str, Any], scores: Dict[str, Any]) -> Dict[str, Any]:
    v = req.vehicle
    prof = enriched.get("profile", {}) or {}
    seg_code = (enriched.get("segment", {}) or {}).get("code", "C_SEDAN")

    yearly_km = int(prof.get("yearly_km") or 15000)
    usage = str(prof.get("usage") or "mixed")
    fuel_pref = str(prof.get("fuel_preference") or "").lower()
    family = str(prof.get("family") or "unknown")
    budget_sens = str(prof.get("budget_sensitivity") or "medium")
    priority = str(prof.get("priority") or "balance")

    vehicle_fuel = str(v.fuel or fuel_pref or "").lower()

    fit = 70

    # yakıt uyumu
    if vehicle_fuel and fuel_pref and vehicle_fuel == fuel_pref:
        fit += 4

    # km & dizel uyumu
    if yearly_km >= 25000:
        if vehicle_fuel == "diesel":
            fit += 8
        if vehicle_fuel in ("hybrid", "electric"):
            fit += 3
    elif yearly_km <= 8000:
        if vehicle_fuel == "diesel":
            fit -= 8
        if vehicle_fuel == "lpg":
            fit -= 2

    # kullanım tipi
    if usage == "city":
        if vehicle_fuel == "diesel":
            fit -= 6
        if vehicle_fuel in ("hybrid", "electric"):
            fit += 5
    elif usage == "highway":
        if vehicle_fuel == "diesel":
            fit += 4

    # bütçe hassasiyeti + premium
    if seg_code in ("PREMIUM_D", "E_SEGMENT"):
        fit -= 3
        if budget_sens == "high":
            fit -= 8

    # aile -> SUV/C segment pratikliği (basit)
    if family == "kids":
        if seg_code in ("C_SUV", "D_SEDAN"):
            fit += 3
        elif seg_code == "B_HATCH":
            fit -= 2

    # öncelik -> maliyet dengesi etkisi
    if priority == "economy":
        cost100 = int(scores.get("cost_balance_100") or 50)
        if cost100 < 45:
            fit -= 6
        elif cost100 > 65:
            fit += 2

    fit = _clamp(fit, 35, 92)

    return {
        "personal_fit_100": int(fit),
        "notes": [
            "Bu skor; yıllık km, kullanım tipi, yakıt tercihi ve segment pratikliği birlikte düşünülerek üretilen göreceli bir uyum göstergesidir."
        ]
    }


# =========================================================
# PERSONALIZED LISTS (short)
# =========================================================
def build_buy_checklist(enriched: Dict[str, Any], req: AnalyzeRequest, max_items: int = 6) -> List[str]:
    risk = enriched.get("risk", {}) or {}
    profile = enriched.get("profile", {}) or {}
    age = risk.get("age")
    mileage = int(risk.get("mileage_km") or 0)
    usage = profile.get("usage", "mixed")

    fuel = (req.vehicle.fuel or req.profile.fuel_preference or "").lower()

    base_list = list(enriched.get("inspection_checklist") or [])
    checklist: List[str] = []

    if (age is not None and age >= 10) or mileage >= 180_000:
        checklist.extend([
            "Motor için kompresyon/yağ kaçak kontrolü ve soğutma sistemi kontrolü yaptır.",
            "Şanzıman geçişlerini (soğuk/sıcak) test sürüşünde dene; adaptasyon/hata kaydı kontrol ettir.",
            "Lift üzerinde alt takım, aks körükleri, salıncak burçları ve direksiyon boşluklarını kontrol ettir.",
        ])
    else:
        checklist.extend([
            "Rutin bakım kalemlerinin (yağ, filtreler, triger, fren) ne zaman değiştiğini belgeyle doğrula.",
            "Lastik diş derinliği, balans ve fren performansını test sürüşünde kontrol et.",
        ])

    if fuel == "diesel":
        checklist.append("Dizel motorda DPF/EGR ve enjektör sağlığını (özellikle şehir içi kullanımsa) kontrol ettir.")
    elif fuel == "lpg":
        checklist.append("LPG montaj/proje/ruhsat uyumu ve subap/kompresyon durumunu kontrol ettir.")
    elif fuel in ("hybrid", "electric"):
        checklist.append("Batarya sağlığı/garanti koşulları ve servis desteğini doğrula.")

    for item in base_list:
        if len(checklist) >= max_items:
            break
        if item not in checklist:
            checklist.append(item)

    # fazla uzamasın
    return checklist[:max_items]


# =========================================================
# PREMIUM TEMPLATE (deterministic) + OPTIONAL LLM REWRITE
# =========================================================
def build_premium_template(req: AnalyzeRequest, enriched: Dict[str, Any]) -> Dict[str, Any]:
    v = req.vehicle
    title = f"{v.year or ''} {v.make} {v.model}".strip() or "Premium Analiz"

    costs = enriched["costs"]
    ins = enriched.get("insurance", {})
    taxes = enriched.get("taxes", {})
    fixed = enriched.get("fixed_costs", {})
    risk = enriched.get("risk", {})
    prof = enriched.get("profile", {})
    idx = (enriched.get("market", {}) or {}).get("indices", {}) or {}

    listed = costs.get("listed_price_try")
    maint_min, maint_max = costs["maintenance_yearly_try_min"], costs["maintenance_yearly_try_max"]
    fuel_mid = costs.get("yearly_fuel_tr_mid")

    mtv = (taxes.get("mtv") or {})
    insp = (fixed.get("inspection") or {})
    traffic = (ins.get("traffic") or {})
    kasko = (ins.get("kasko") or {})

    parts_av = int(idx.get("parts_availability_score_1_5", 3) or 3)
    parts_cost = int(idx.get("parts_cost_index_1_5", 3) or 3)
    service_net = int(idx.get("service_network_index_1_5", 3) or 3)
    resale_liq = int(idx.get("resale_liquidity_score_1_5", 3) or 3)

    # scores
    s = _compute_overall_score(req, enriched)
    fit = _compute_personal_fit(req, enriched, s)

    overall_100 = int(s["overall_100"])
    mech_100 = int(s["mechanical_100"])
    elec_100 = int(s["electronics_100"])
    cost_100 = int(s["cost_balance_100"])
    liq_100 = int(s["liquidity_100"])
    fit_100 = int(fit["personal_fit_100"])

    overall_10 = _to10(overall_100)
    mech_10 = _to10(mech_100)
    elec_10 = _to10(elec_100)
    cost_10 = _to10(cost_100)
    liq_10 = _to10(liq_100)
    fit_10 = _to10(fit_100)

    # checklist
    checklist = build_buy_checklist(enriched, req, max_items=6)

    # ---- result text (premium, structured) ----
    lines: List[str] = []
    lines.append("## Premium Analiz Sonucu")
    lines.append(f"Genel Puan: {overall_100} / 100")
    lines.append("")
    lines.append("### Detaylı Değerlendirme")
    lines.append(f"## {title}")
    lines.append("")

    iq = enriched.get("info_quality", {})
    lines.append(f"**Bilgi seviyesi:** {iq.get('level','-')} (eksikler: {', '.join(iq.get('missing_fields', [])[:6]) or 'yok'})")
    lines.append("")

    # 1) skor
    lines.append("### 1) Oto Analiz Skoru")
    lines.append(f"**{overall_10} / 10** _(100’lük karşılığı: {overall_100})_")
    lines.append("")

    # 2) alt skorlar
    lines.append("### 2) Alt Skorlar")
    lines.append(f"- Mekanik: **{mech_10} / 10**")
    lines.append(f"- Elektronik/Donanım: **{elec_10} / 10**")
    lines.append(f"- Maliyet Dengesi: **{cost_10} / 10**")
    lines.append(f"- 2. El Likidite: **{liq_10} / 10**")
    lines.append("")

    # 3) kişiye uygunluk
    prof_line = f"{prof.get('yearly_km','-')} km/yıl, kullanım: {prof.get('usage','-')}, tercih: {prof.get('fuel_preference','-')}"
    seg_label = enriched.get("segment", {}).get("name", "segment")
    lines.append("### 3) Kişiye Uygunluk")
    lines.append(f"**{fit_10} / 10** _(100’lük karşılığı: {fit_100})_")
    lines.append(f"- Profil: **{prof_line}**")
    lines.append(f"- Segment: **{seg_label}**")
    lines.append("Bu skor; kullanım alışkanlığı, yıllık km bandı, yakıt tercihi ve segment pratikliği birlikte düşünülerek üretilmiş göreceli bir uyum göstergesidir.")
    lines.append("")

    # 4) maliyet
    lines.append("### 4) Yıllık maliyet özeti (tahmini band)")
    lines.append(f"- Bakım: **{_fmt_try(maint_min)} – {_fmt_try(maint_max)} TL/yıl**")
    lines.append(f"  - Periyodik pay: ~**{_fmt_try(costs['maintenance_routine_yearly_est'])} TL/yıl**")
    lines.append(f"  - Beklenmedik gider payı: ~**{_fmt_try(costs['maintenance_risk_reserve_yearly_est'])} TL/yıl**")
    lines.append(f"- Yakıt/enerji: **{_fmt_try(costs['yearly_fuel_tr_min'])} – {_fmt_try(costs['yearly_fuel_tr_max'])} TL/yıl** (orta: {_fmt_try(fuel_mid)} TL)")
    if mtv.get("ok"):
        lines.append(f"- MTV: **{_fmt_try(mtv['mtv_yearly_try_min'])} – {_fmt_try(mtv['mtv_yearly_try_max'])} TL/yıl**")
    if insp.get("ok"):
        lines.append(f"- Muayene yıllık ortalama: **~{_fmt_try(insp['annual_equivalent_try_2026_estimated'])} TL/yıl**")
    if traffic.get("ok"):
        lines.append(f"- Trafik sigortası: **{_fmt_try(traffic['traffic_est_try_min'])} – {_fmt_try(traffic['traffic_est_try_max'])} TL/yıl**")
    if kasko.get("ok"):
        lines.append(f"- Kasko: **{_fmt_try(kasko['kasko_try_min'])} – {_fmt_try(kasko['kasko_try_max'])} TL/yıl**")
    lines.append(f"- Not: Yakıt türü + kullanım şekli bazı kalemlerin takibini sıklaştırabilir. ({risk.get('fuel_comment','-')})")
    lines.append("")

    # 5) endeksler
    lines.append("### 5) Parça, servis ve ikinci el endeksleri")
    lines.append(f"- Parça bulunabilirliği: **{parts_av}/5**")
    lines.append(f"- Parça maliyet endeksi: **{parts_cost}/5**")
    lines.append(f"- Servis ağı: **{service_net}/5**")
    lines.append(f"- 2. el likidite: **{resale_liq}/5**")
    lines.append("")

    # 6) kontrol listesi
    lines.append("### 6) Kontrol listesi (kısa)")
    for c in checklist[:6]:
        lines.append(f"- {c}")
    lines.append("")

    # 7) notlar
    lines.append("### 7) Notlar")
    lines.append("Bu rapor; segment emsalleri, yaş/km ve kullanım profiline göre **band** üretir.")
    lines.append("İlan detayları arttıkça bandlar daralır ve skorlar daha isabetli hale gelir.")
    lines.append("")

    # artılar / eksiler
    lines.append("**Artılar**")
    lines.append("- Genel ve alt skorlar + kişiye uygunluk birlikte verildi.")
    lines.append("- Yıllık maliyet; bakım/yakıt/sigorta/vergi kalemlerine ayrıldı.")
    lines.append("- Parça/servis/likidite endeksleri değerlendirmeye dahil edildi.")
    lines.append("")
    lines.append("**Eksiler**")
    lines.append("- Tramer ve bakım geçmişi netleşmeden skorlar daha geniş belirsizlik bandında kalabilir.")
    lines.append("- Kasko ve sigorta teklifleri sürücü geçmişine göre ciddi değişebilir.")
    lines.append("")

    # SON ÖZET (en sonda)
    lines.append("### 8) Son Özet")
    lines.append(f"- Skor özeti: **{overall_10}/10** genel; mekanik **{mech_10}/10**, elektronik **{elec_10}/10**, maliyet **{cost_10}/10**, likidite **{liq_10}/10**, kişiye uygunluk **{fit_10}/10**.")
    lines.append(f"- Profil: **{prof_line}** → skorlar bu senaryoya göre yorumlandı.")
    lines.append(f"- Maliyet bandı: bakım **{_fmt_try(maint_min)}–{_fmt_try(maint_max)}**, yakıt **{_fmt_try(costs['yearly_fuel_tr_min'])}–{_fmt_try(costs['yearly_fuel_tr_max'])}**; sigorta/MTV/kasko kalemleri yukarıda ayrı verildi.")
    lines.append("- Belirsizliği azaltan 3 veri: **tramer detayı**, **bakım/fatura kayıtları**, **OBD + test sürüşü**. Bu bilgiler geldikçe band daralır ve skorlar daha isabetli olur.")

    result_text = "\n".join(lines)
    result_text = _sanitize_no_verdict(result_text)

    # preview tag (rakam yok)
    price_tag = None
    if listed:
        if listed < 500_000:
            price_tag = "Uygun"
        elif listed < 1_200_000:
            price_tag = "Normal"
        else:
            price_tag = "Yüksek"

    out = {
        "scores": {
            "overall_100": overall_100,
            "overall_10": overall_10,

            "mechanical_100": mech_100,
            "mechanical_10": mech_10,

            "electronics_100": elec_100,
            "electronics_10": elec_10,

            "cost_balance_100": cost_100,
            "cost_balance_10": cost_10,

            "liquidity_100": liq_100,
            "liquidity_10": liq_10,

            "personal_fit_100": fit_100,
            "personal_fit_10": fit_10,
        },
        "summary": {
            "short_comment": "Skorlar ve maliyet bandı; yaş/km, segment emsalleri ve kullanım profiline göre üretildi. İlan detayları arttıkça band daralır.",
            "pros": [
                "Skorlar (genel + alt skorlar) ve kişiye uygunluk birlikte verildi.",
                "Yıllık maliyet bandı kalemlere ayrıldı (bakım/yakıt/sigorta/vergi).",
                "Parça/servis/likidite endeksleri özetlendi."
            ],
            "cons": [
                "Tramer ve bakım geçmişi netleşmeden belirsizlik bandı geniş kalabilir.",
                "Kasko/sigorta teklifleri sürücü profiline göre ciddi değişebilir.",
                "İlan açıklaması kısa ise bazı alanlar daha genel kalır."
            ],
            "who_should_buy": "Bu rapor; karar yerine skor ve band verir. Ekspertiz + tramer + OBD ile desteklendiğinde daha net ve isabetli hale gelir.",
        },
        "preview": {
            "title": title,
            "price_tag": price_tag,
            "spoiler": "Skorlar + kişiye uygunluk + yıllık maliyet bandı + kontrol listesi.",
            "bullets": [
                "Genel ve alt skorlar",
                "Kişiye uygunluk skoru",
                "Bakım/yakıt/sigorta/MTV bandı",
                "Kısa kontrol listesi",
            ],
        },
        "details": {
            "indices": {
                "parts_availability_1_5": parts_av,
                "parts_cost_1_5": parts_cost,
                "service_network_1_5": service_net,
                "resale_liquidity_1_5": resale_liq,
            },
            "profile_used": prof,
            "segment": enriched.get("segment"),
            "info_quality": enriched.get("info_quality"),
        },
        "fixed_costs": {
            "mtv": mtv,
            "inspection": insp,
            "traffic": traffic,
            "kasko": kasko,
        },
        "result": result_text,
    }
    return out


# =========================================================
# LLM HELPERS (optional) - premium tone rewrite
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
Sen Oto Analiz uygulaması için PREMIUM analiz metinlerini 'premium danışman' tonunda iyileştiren bir asistansın.

Sadece şunları iyileştir:
- summary.short_comment
- summary.pros (3-5 madde)
- summary.cons (3-5 madde)
- summary.who_should_buy (1-3 cümle)
- result (kullanıcıya gösterilen ana metin)

Kurallar:
- Rakamları UYDURMA, verilen sayıları değiştirme. (Skorlar ve TL değerleri aynı kalmalı.)
- 'alınır/alınmaz', 'sakın', 'tehlikeli', 'riskli' gibi kesin/etiketleyici ifadeler kullanma.
- Metin ansiklopedi/Google gibi değil; doğal, net, aksiyon odaklı olsun.
- result içinde başlıkları KORU, sadece dili güzelleştir.
- result metninin EN SONUNDA mutlaka "### 8) Son Özet" başlığı ve kısa kapanış maddeleri olmalı.

ÇIKTI SADECE JSON:
{
  "summary": {"short_comment":"", "pros":[], "cons":[], "who_should_buy":""},
  "result": ""
}
""".strip()


def premium_analyze_impl(req: AnalyzeRequest) -> Dict[str, Any]:
    enriched = build_enriched_context(req)
    base = build_premium_template(req, enriched)

    # LLM varsa: summary + result rewrite (rakamları değiştirmeden)
    rewrite_input = {
        "vehicle": req.vehicle.dict(),
        "profile": req.profile.dict(),
        "info_quality": enriched.get("info_quality"),
        "segment": enriched.get("segment"),
        "base_summary": base.get("summary"),
        "base_result": base.get("result"),
        "banned_words": ["alınır", "alınmaz", "sakın", "tehlikeli", "riskli"],
    }

    llm = call_llm_json(
        model_name=OPENAI_MODEL_PREMIUM,
        system_prompt=SYSTEM_PROMPT_PREMIUM_REWRITE,
        user_content=json.dumps(rewrite_input, ensure_ascii=False),
    )

    if isinstance(llm, dict):
        try:
            if "summary" in llm and isinstance(llm["summary"], dict):
                base["summary"].update({
                    k: llm["summary"][k]
                    for k in ["short_comment", "pros", "cons", "who_should_buy"]
                    if k in llm["summary"]
                })

            if "result" in llm and isinstance(llm["result"], str) and llm["result"].strip():
                base["result"] = _sanitize_no_verdict(llm["result"].strip())
        except Exception:
            pass

    # final sanitize
    base["result"] = _sanitize_no_verdict(base.get("result", ""))

    return base


# =========================================================
# NORMAL / MANUAL / COMPARE / OTOBOT (light LLM)
# =========================================================
SYSTEM_PROMPT_NORMAL = """
Sen 'Oto Analiz' uygulaması için çalışan bir araç ilanı analiz asistanısın.
ÇIKTIYI SADECE GEÇERLİ BİR JSON OLARAK DÖN. ŞABLON:
{
  "scores": {"overall_100":0,"mechanical_100":0,"body_100":0,"economy_100":0},
  "summary": {"short_comment":"","pros":[],"cons":[],"estimated_risk_level":"orta"},
  "preview": {"title":"","price_tag":null,"spoiler":"","bullets":[]},
  "result": ""
}
Kurallar:
- 'alınır/alınmaz/sakın/tehlikeli/riskli' kullanma.
- PREVIEW price_tag: sadece 'Uygun'/'Normal'/'Yüksek' veya null, rakam yok.
- Dil Türkçe, sadece JSON.
""".strip()


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
Dil Türkçe, sadece JSON.
""".strip()


SYSTEM_PROMPT_OTOBOT = """
Çıktı sadece JSON:
{"answer":"","suggested_segments":[],"example_models":[]}
Dil Türkçe, sadece JSON.
""".strip()


def fallback_normal(req: AnalyzeRequest) -> Dict[str, Any]:
    v = req.vehicle
    title = f"{v.year or ''} {v.make} {v.model}".strip() or "Araç Analizi"
    return {
        "scores": {"overall_100": 70, "mechanical_100": 70, "body_100": 70, "economy_100": 70},
        "summary": {
            "short_comment": "Sınırlı bilgiye göre genel bir değerlendirme yapıldı.",
            "pros": ["Detaylar ekspertiz ve tramer ile netleştiğinde daha isabetli olur."],
            "cons": ["Bakım geçmişi ve hasar kaydı netleşmeden yorumlar daha genel kalabilir."],
            "estimated_risk_level": "orta",
        },
        "preview": {
            "title": title,
            "price_tag": None,
            "spoiler": "Genel değerlendirme hazır. Ekspertiz ve bakım geçmişi teyit edilmeden karar verilmemeli.",
            "bullets": ["Tramer/hasar kaydı", "Bakım kayıtları", "Test sürüşü + OBD"],
        },
        "result": "Genel bir değerlendirme sağlandı. Kesinleşmesi için ilan detayları, ekspertiz raporu ve tramer sorgusu birlikte değerlendirilmelidir.",
    }


def call_llm_or_fallback(model_name: str, system_prompt: str, user_content: str, fallback_fn, req_obj):
    out = call_llm_json(model_name, system_prompt, user_content)
    if isinstance(out, dict):
        # ekstra sanitize
        if "result" in out and isinstance(out["result"], str):
            out["result"] = _sanitize_no_verdict(out["result"])
        return out
    return fallback_fn(req_obj)


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
    v = req.vehicle
    p = req.profile
    user_content = {
        "mode": "normal",
        "vehicle": v.dict(),
        "profile": p.dict(),
        "listed_price_try": _parse_listed_price(req),
        "ad_description": req.ad_description or "",
        "hint": "Kesin hüküm verme; skor ve dikkat noktalarıyla dengeli bir özet üret. Yasak kelimeler: alınır/alınmaz/sakın/tehlikeli/riskli."
    }
    return call_llm_or_fallback(
        OPENAI_MODEL_NORMAL,
        SYSTEM_PROMPT_NORMAL,
        json.dumps(user_content, ensure_ascii=False),
        fallback_normal,
        req
    )


@app.post("/premium_analyze")
async def premium_analyze(req: AnalyzeRequest) -> Dict[str, Any]:
    return premium_analyze_impl(req)


@app.post("/manual_analyze")
@app.post("/manual")
async def manual_analyze(req: AnalyzeRequest) -> Dict[str, Any]:
    v = req.vehicle
    p = req.profile
    user_content = {
        "mode": "manual",
        "vehicle": v.dict(),
        "profile": p.dict(),
        "listed_price_try": _parse_listed_price(req),
        "ad_description": req.ad_description or "",
        "hint": "Kesin hüküm verme; kullanıcı aracı olabilir. Bakım planı ve dikkat noktalarıyla pratik özet üret."
    }
    return call_llm_or_fallback(
        OPENAI_MODEL_NORMAL,
        SYSTEM_PROMPT_NORMAL,
        json.dumps(user_content, ensure_ascii=False),
        fallback_normal,
        req
    )


@app.post("/compare_analyze")
async def compare_analyze(req: CompareRequest) -> Dict[str, Any]:
    left_v = req.left.vehicle
    right_v = req.right.vehicle
    payload = {
        "left": {"vehicle": left_v.dict(), "ad_description": req.left.ad_description or ""},
        "right": {"vehicle": right_v.dict(), "ad_description": req.right.ad_description or ""},
        "profile": req.profile.dict() if req.profile else None,
    }
    out = call_llm_json(OPENAI_MODEL_COMPARE, SYSTEM_PROMPT_COMPARE, json.dumps(payload, ensure_ascii=False))
    if isinstance(out, dict):
        return out
    return {
        "better_overall": "left",
        "summary": "Kıyas; ilan bilgileri sınırlıysa genelde kalır. Ekspertiz/tramer/OBD ile netleşir.",
        "left_pros": ["Genel denge daha iyi olabilir."],
        "left_cons": ["Detaylar (hasar/bakım) netleşmeden fark kapanabilir."],
        "right_pros": ["Doğru bakım geçmişiyle iyi bir alternatif olabilir."],
        "right_cons": ["Masraf kalemleri farklılaşabilir; detaylı kontrol gerekir."],
        "use_cases": {
            "family_use": "Aile kullanımında hacim/konfor ve yakıt maliyeti belirleyicidir.",
            "long_distance": "Uzun yolda bakım disiplini ve tüketim öne çıkar.",
            "city_use": "Şehir içinde şanzıman tipi ve dur-kalk yıpranması belirleyicidir."
        }
    }


@app.post("/otobot")
async def otobot(req: OtoBotRequest) -> Dict[str, Any]:
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Soru boş olamaz. 'question' alanına bir metin gönder.")
    out = call_llm_json(OPENAI_MODEL_OTOBOT, SYSTEM_PROMPT_OTOBOT, question)
    if isinstance(out, dict):
        return out
    return {
        "answer": "Bütçe, yıllık km ve kullanım tipine göre segment seçmek en mantıklısıdır. İlanları kısa listeleyip ekspertiz+tramer ile netleştirmek gerekir.",
        "suggested_segments": ["C-sedan", "C-SUV", "B-Hatch"],
        "example_models": ["Toyota Corolla", "Renault Megane", "Honda Civic", "Hyundai Tucson"]
    }
