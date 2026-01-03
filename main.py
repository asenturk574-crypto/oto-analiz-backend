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


class Vehicle(BaseModel):
    make: str = ""
    model: str = ""
    year: Optional[int] = Field(None, ge=1980, le=2035)
    mileage_km: Optional[int] = Field(None, ge=0)
    fuel: Optional[str] = None        # gasoline / diesel / lpg / hybrid / electric


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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _load_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return default


def _dp(path_in_data: str) -> str:
    # DATA_DIR env ile override edilebiliyor; relative ise main.py'nin bulunduğu klasöre göre çöz
    data_dir = DATA_DIR
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(BASE_DIR, data_dir)
    return os.path.join(data_dir, path_in_data)


def _merge_by_key_prefer_first(primary: List[Dict[str, Any]], secondary: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    primary + secondary birleştirir.
    - Aynı 'key' varsa primary (örn. v2) kazanır
    - key yoksa JSON-serialize ile kaba dedupe yapar
    """
    out: List[Dict[str, Any]] = []
    seen_keys = set()
    seen_fallback = set()

    def k_of(d: Dict[str, Any]) -> Optional[str]:
        k = d.get("key")
        if not k:
            return None
        try:
            return _norm(str(k))
        except:
            return None

    def fallback_sig(d: Dict[str, Any]) -> str:
        try:
            return json.dumps(d, ensure_ascii=False, sort_keys=True)
        except:
            return str(d)

    for src in (primary or []):
        if not isinstance(src, dict):
            continue
        k = k_of(src)
        if k:
            if k in seen_keys:
                continue
            seen_keys.add(k)
            out.append(src)
        else:
            sig = fallback_sig(src)
            if sig in seen_fallback:
                continue
            seen_fallback.add(sig)
            out.append(src)

    for src in (secondary or []):
        if not isinstance(src, dict):
            continue
        k = k_of(src)
        if k:
            if k in seen_keys:
                continue
            seen_keys.add(k)
            out.append(src)
        else:
            sig = fallback_sig(src)
            if sig in seen_fallback:
                continue
            seen_fallback.add(sig)
            out.append(src)

    return out


# --- v2 varsa öncelikli oku (yoksa v1) + ikisini birlikte merge et
ANCHORS_V2: List[Dict[str, Any]] = _load_json(_dp("anchors_tr_popular_v2_227.json"), [])
ANCHORS_V1: List[Dict[str, Any]] = _load_json(_dp("anchors_tr_popular_96.json"), [])
ANCHORS: List[Dict[str, Any]] = _merge_by_key_prefer_first(ANCHORS_V2, ANCHORS_V1)

VEHICLE_PROFILES_V2: List[Dict[str, Any]] = _load_json(_dp("vehicle_profiles_v2_227.json"), [])
VEHICLE_PROFILES_V1: List[Dict[str, Any]] = _load_json(_dp("vehicle_profiles_96_v1.json"), [])
VEHICLE_PROFILES: List[Dict[str, Any]] = _merge_by_key_prefer_first(VEHICLE_PROFILES_V2, VEHICLE_PROFILES_V1)

# Diğer pack dosyaları aynı isimle zaten güncellendiyse otomatik onları okur (replace yaptıysan yeni içerik okunur)
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
            "Parça ve usta erişimi genelde kolaydır; şehir içi kullanım için mantıklı bir sınıftır.",
            "Doğru bakım yapıldığında yıllık masraflar çoğu zaman kontrol edilebilir kalır.",
        ],
    },
    "C_SEDAN": {
        "name": "C segment (aile sedan/hatchback)",
        "maintenance_yearly_range": (15000, 32000),
        "insurance_level": "orta",
        "notes": [
            "Türkiye’de en likit segmentlerden biridir; temiz örneklerin alıcısı genelde vardır.",
            "Konfor/donanım artarken bakım maliyeti B segmente göre bir miktar yükselir.",
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
            "Konfor ve donanım yüksek; parça/işçilik maliyetleri C segmente göre belirgin artabilir.",
        ],
    },
    "PREMIUM_D": {
        "name": "Premium D segment",
        "maintenance_yearly_range": (32000, 75000),
        "insurance_level": "yüksek",
        "notes": [
            "Premium sınıfta işçilik, bakım ve parça maliyeti belirgin yüksektir.",
            "Elektronik/donanım arızaları yaşla birlikte daha sık görülebilir.",
        ],
    },
    "E_SEGMENT": {
        "name": "E segment / üst sınıf",
        "maintenance_yearly_range": (45000, 120000),
        "insurance_level": "çok yüksek",
        "notes": [
            "Üst sınıf büyük gövdeli araçlarda masraf kalemleri bariz şekilde yüksektir.",
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


# =========================================================
# COST + RISK ESTIMATION (band)
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
        "note": "Trafik sigortası bandı, il ve basamak tavanına göre tahmini verildi; gerçek teklif sürücü/hasar geçmişine göre değişir.",
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
        "note": "Kasko bandı araç değeri + segment + yaş/km ile tahmini verildi. Şehir, teminatlar, sürücü profili ve hasarsızlık indirimiyle ciddi değişir.",
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
    risk_reserve_est = max(0, mid_maint - routine_est)

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
        risk_notes.append("Km çok yüksek; motor/şanzıman revizyon riski artmış olabilir.")
    elif mileage >= 180_000 and risk_level != "yüksek":
        risk_level = "orta-yüksek"
        risk_notes.append("Km yüksek; yürüyen aksam ve mekanik masraf ihtimali artmış olabilir.")

    if "yüksek" in fuel_risk:
        risk_level = "yüksek"

    if segment_code in ("PREMIUM_D", "E_SEGMENT") and ((age and age > 10) or mileage > 180_000):
        risk_notes.append("Premium sınıfta yaşlı/yüksek km araçların büyük masraf kalemleri pahalı olabilir.")

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
        "maintenance_risk_reserve_yearly_est": risk_reserve_est,
        "yearly_fuel_tr_min": fuel_min,
        "yearly_fuel_tr_max": fuel_max,
        "yearly_fuel_tr_mid": fuel_mid,
        "insurance_level": seg["insurance_level"],
        "insurance": {"traffic": traffic, "kasko": kasko},
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
            "insurance_level": seg_info["insurance_level"],
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
            "fuel_risk_comment": seg_info["fuel_risk_comment"],
            "baseline_risk_level": seg_info["risk_level"],
            "risk_notes": seg_info["risk_notes"],
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
        },
        "info_quality": info_q,
        "anchors_used": [
            {"key": a.get("key"), "segment": a.get("segment"), "tags": a.get("tags", []), "parts": a.get("parts"), "resale": a.get("resale")}
            for a in anchors
        ],
        "inspection_checklist": checklist[:10],
        "tags_detected": guess_tags,
    }


# =========================================================
# PREMIUM: EXTRA "DOLU DOLU" HELPERS (FLAGS / INDEX EXPLAIN / UNCERTAINTY / VALUE)
# =========================================================
def _score_from_risk(risk_level: str) -> int:
    if risk_level == "düşük":
        return 82
    if risk_level == "orta":
        return 76
    if risk_level == "orta-yüksek":
        return 70
    if risk_level == "yüksek":
        return 62
    return 74


def _level_to_uncertainty(level: str) -> Tuple[str, int]:
    if level == "yüksek":
        return ("düşük", 22)
    if level == "orta":
        return ("orta", 45)
    return ("yüksek", 68)


def build_uncertainty(enriched: Dict[str, Any]) -> Dict[str, Any]:
    iq = enriched.get("info_quality", {}) or {}
    match = (enriched.get("market", {}) or {}).get("profile_match")
    missing = iq.get("missing_fields", []) or []

    base_level, base_score = _level_to_uncertainty(iq.get("level", "orta"))

    if match == "segment_estimate":
        base_score += 12

    base_score += min(18, 3 * len(missing))

    score = _clamp(int(base_score), 0, 100)
    if score <= 30:
        level = "düşük"
    elif score <= 55:
        level = "orta"
    else:
        level = "yüksek"

    improve = []
    if "ilan_aciklamasi_kisa" in missing:
        improve.append("İlan açıklaması kısa: bakım geçmişi, değişen/boya, şanzıman tipi ve tramer bilgisi netleşirse rapor daralır.")
    if "km" in missing:
        improve.append("Km bilgisi netleşirse maliyet/risk bandı daha doğru olur.")
    if "yil" in missing:
        improve.append("Model yılı netleşirse vergi ve yaş kaynaklı riskler daha doğru hesaplanır.")
    if match == "segment_estimate":
        improve.append("Bu model için direkt profil yok: paket/motor/şanzıman bilgisi gelirse emsal seçimi iyileşir.")

    if not improve:
        improve.append("Tramer + servis kayıtları + OBD taraması raporu en çok netleştirir.")

    return {
        "level": level,
        "score_100": score,
        "missing_fields": missing[:8],
        "profile_match": match,
        "how_to_improve": improve[:4],
    }


def _fit_score(req: AnalyzeRequest, enriched: Dict[str, Any]) -> int:
    prof = enriched.get("profile", {}) or {}
    seg = (enriched.get("segment", {}) or {}).get("code", "C_SEDAN")
    fuel = (req.vehicle.fuel or req.profile.fuel_preference or "").lower()

    score = 72

    if prof.get("usage") == "city" and fuel == "diesel" and prof.get("yearly_km", 15000) < 12000:
        score -= 10

    if prof.get("yearly_km", 15000) >= 30000:
        if fuel in ("diesel", "hybrid"):
            score += 5
        score += 2

    if seg in ("PREMIUM_D", "E_SEGMENT") and prof.get("yearly_km_band") == "düşük":
        score -= 6

    if seg in ("C_SEDAN", "C_SUV", "D_SEDAN"):
        score += 3

    return _clamp(int(score), 0, 100)


def _electronics_score(enriched: Dict[str, Any]) -> int:
    risk = enriched.get("risk", {}) or {}
    seg = (enriched.get("segment", {}) or {}).get("code", "C_SEDAN")
    age = risk.get("age") or 0

    score = 74
    if seg in ("PREMIUM_D", "E_SEGMENT"):
        score -= 5
    if age >= 10:
        score -= 6
    if age >= 15:
        score -= 6
    return _clamp(int(score), 0, 100)


def build_flags(enriched: Dict[str, Any], req: AnalyzeRequest) -> Dict[str, List[str]]:
    risk = enriched.get("risk", {}) or {}
    prof = enriched.get("profile", {}) or {}
    uq = enriched.get("info_quality", {}) or {}
    seg = (enriched.get("segment", {}) or {}).get("code", "C_SEDAN")

    age = risk.get("age")
    km = int(risk.get("mileage_km") or 0)
    fuel = (req.vehicle.fuel or req.profile.fuel_preference or "").lower()
    usage = prof.get("usage", "mixed")

    red: List[str] = []
    yellow: List[str] = []
    green: List[str] = []

    if age is not None and age >= 15:
        red.append("Yaş yüksek → büyük bakım/elektronik masraf ihtimali artar. (Aksiyon: servis geçmişi + detaylı ekspertiz + OBD)")
    if km >= 250_000:
        red.append("Km çok yüksek → motor/şanzıman/alt takım masraf bandı büyür. (Aksiyon: test sürüşü + kompresyon/kaçak kontrolü)")
    if fuel == "diesel" and usage == "city" and km >= 120_000:
        red.append("Şehir içi ağırlıklı dizel → DPF/EGR/enjektör riski artabilir. (Aksiyon: DPF doluluk/arıza kodu kontrolü)")
    if seg in ("PREMIUM_D", "E_SEGMENT") and ((age or 0) >= 10 or km >= 180_000):
        red.append("Premium + yaş/km yükseldikçe parça/işçilik maliyeti çarpan etkisi yapar. (Aksiyon: yedek bütçe + detaylı kontrol)")

    if uq.get("level") in ("düşük", "orta"):
        yellow.append("İlan/araç bilgileri kısmi → skor bandı daha geniş yorumlanmalı. (Aksiyon: tramer + bakım faturası + net paket/motor)")
    if (enriched.get("market", {}) or {}).get("profile_match") == "segment_estimate":
        yellow.append("Bu model için direkt profil yok → emsal segmentten tahmin kullanıldı. (Aksiyon: motor/şanzıman/paket netleşirse iyileşir)")
    if "lpg" in fuel:
        yellow.append("LPG varsa montaj/ayar kalitesi belirleyici. (Aksiyon: kompresyon + subap durumu + proje/ruhsat kontrolü)")
    if fuel in ("hybrid", "electric"):
        yellow.append("Hibrit/EV tarafında batarya sağlığı kritik. (Aksiyon: batarya raporu/garanti şartlarını doğrula)")

    if age is not None and age <= 6 and km <= 120_000:
        green.append("Yaş/km görece dengeli → büyük masraf olasılığı daha yönetilebilir. (Yine de ekspertiz şart)")
    if seg in ("C_SEDAN", "C_SUV"):
        green.append("Segment yaygın → parça/usta erişimi genelde daha rahat olur.")
    if prof.get("yearly_km", 15000) >= 20000 and fuel in ("diesel", "hybrid"):
        green.append("Yıllık km yüksekse yakıt ekonomisi avantajı daha anlamlı hale gelir.")
    if prof.get("usage") == "highway":
        green.append("Uzun yol ağırlıklı kullanım → tüketim daha stabil olabilir; bakım disiplini değerli olur.")

    return {
        "red": red[:3],
        "yellow": yellow[:3],
        "green": green[:3],
    }


def explain_indices(enriched: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    seg = (enriched.get("segment", {}) or {}).get("code", "C_SEDAN")
    match = (enriched.get("market", {}) or {}).get("profile_match")
    idx = (enriched.get("market", {}) or {}).get("indices", {}) or {}

    parts_av = int(idx.get("parts_availability_score_1_5", 3))
    parts_cost = int(idx.get("parts_cost_index_1_5", 3))
    service_net = int(idx.get("service_network_index_1_5", 3))
    resale_liq = int(idx.get("resale_liquidity_score_1_5", 3))

    def _why_common(topic: str) -> List[str]:
        reasons = []
        if match == "segment_estimate":
            reasons.append("Bu puanlar bu modelin direkt verisi değil; aynı segment emsallerinin ortalamasından tahmini türetildi.")
        if seg in ("PREMIUM_D", "E_SEGMENT"):
            reasons.append("Premium/üst sınıfta OEM parça ve işçilik maliyeti genelde daha yüksek olur; bazı parçalarda termin etkilenebilir.")
        else:
            reasons.append("Yaygın segmentlerde muadil parça/usta ağı daha güçlü olma eğilimindedir.")
        return reasons

    def _meaning(score: int, kind: str) -> str:
        if kind == "availability":
            if score <= 2:
                return "Parça bulunurluğu daha sınırlı olabilir (stok/termin riski)."
            if score == 3:
                return "Ortalama bulunurluk; kritik parçalarda termin değişebilir."
            return "Parça bulunurluğu genelde iyi; muadil seçenekler daha rahat bulunabilir."
        if kind == "cost":
            if score <= 2:
                return "Parça maliyeti görece düşük; rutin kalemlerde bütçe daha rahat olabilir."
            if score == 3:
                return "Ortalama maliyet; parça/işçilik araca göre değişir."
            return "Parça/işçilik maliyeti yüksek eğilimli; özellikle elektronik ve gövde parçaları pahalı olabilir."
        if kind == "service":
            if score <= 2:
                return "Uzman usta/servis bulma daha zor olabilir; doğru servis seçimi kritik."
            if score == 3:
                return "Ortalama servis erişimi; şehir/bölgeye göre değişir."
            return "Servis/usta ağı geniş; alternatif servis bulmak daha kolay olabilir."
        if kind == "resale":
            if score <= 2:
                return "2. el satış hızı görece yavaş olabilir; doğru fiyatlama önemli."
            if score == 3:
                return "Ortalama likidite; temiz geçmiş + doğru paket hızlı satışı destekler."
            return "2. el likiditesi yüksek; temiz örneklerin alıcısı daha hızlı bulunabilir."
        return "Ortalama."

    out = {
        "parts_availability": {
            "score_1_5": parts_av,
            "meaning": _meaning(parts_av, "availability"),
            "why": _why_common("availability")[:3],
            "action": "Aksiyon: kritik parçalar (far/airbag/elektronik modül) için tedarik süresi ve muadil seçenekleri servisle konuş."
        },
        "parts_cost": {
            "score_1_5": parts_cost,
            "meaning": _meaning(parts_cost, "cost"),
            "why": _why_common("cost")[:3],
            "action": "Aksiyon: bakım kalemlerinde (fren/lastik/yağ) fiyat araştırması yap; OEM/muadil farkını öğren."
        },
        "service_network": {
            "score_1_5": service_net,
            "meaning": _meaning(service_net, "service"),
            "why": _why_common("service")[:3],
            "action": "Aksiyon: bulunduğun ilde 2–3 uzman servis seçip ekspertiz/OBD için fiyat al."
        },
        "resale_liquidity": {
            "score_1_5": resale_liq,
            "meaning": _meaning(resale_liq, "resale"),
            "why": [
                "Likidite; talep genişliği, paket/donanım popülerliği, geçmiş temizliği ve fiyatlamaya çok bağlıdır.",
                "Aynı modelin ilan sayısı fazlaysa doğru fiyatlama daha kritik hale gelir.",
            ][:2] + (["Bu skor segment emsallerinin gözlemsel eğilimini temsil eder."] if match == "segment_estimate" else []),
            "action": "Aksiyon: aynı segmentte 3–5 emsal ilanı (km/hasar/paket) kıyaslayıp fiyatlama mantığını kontrol et."
        },
    }
    return out


def build_value_and_negotiation(enriched: Dict[str, Any]) -> Dict[str, Any]:
    costs = enriched.get("costs", {}) or {}
    listed = costs.get("listed_price_try")
    segment_code = (enriched.get("segment", {}) or {}).get("code", "C_SEDAN")
    iq = enriched.get("info_quality", {}) or {}
    missing = iq.get("missing_fields", []) or []

    if listed is None:
        return {
            "ok": False,
            "note": "İlan fiyatı bulunamadığı için değer/pazarlık bölümü sınırlı tutuldu.",
            "label": None,
            "comment": "Fiyat girilirse (veya context.listed_price_text gelirse) daha net etiket üretilebilir.",
            "negotiation_args": [
                "Ekspertiz raporundaki masrafları kalem kalem pazarlığa çevir.",
                "Bakım kayıtları yoksa yakın vade bakım bütçesini argüman yap.",
            ],
        }

    if listed < 500_000:
        label = "Uygun"
    elif listed < 1_200_000:
        label = "Normal"
    else:
        label = "Yüksek"

    comment = "Değer yorumu; segment, yaş/km ve ilan detay düzeyine göre *yaklaşık* konumlandırma sağlar. Kesin piyasa fiyatı değildir."
    if segment_code in ("PREMIUM_D", "E_SEGMENT"):
        comment += " Premium/üst sınıfta aynı fiyat bandında bile masraf/riske bağlı değer algısı çok değişebilir."
    if "ilan_aciklamasi_kisa" in missing:
        comment += " İlan açıklaması kısa olduğu için pazarlık argümanları daha çok ekspertiz sonucuna dayanmalı."

    args = [
        "Ekspertizde çıkan masrafları (lastik, fren, alt takım, bakım) tek tek fiyatlandırıp pazarlık maddesi yap.",
        "Bakım kayıtları ve faturalar yoksa: yakın vadeli bakım bütçesini (yağ/filtre/sıvılar) argümanlaştır.",
        "Tramer toplamı + parça değişimi bilgilerini (şasi/podye kontrolüyle birlikte) teyit edip belirsizliği pazarlıkta kullan.",
    ]
    return {
        "ok": True,
        "label": label,
        "comment": comment,
        "negotiation_args": args[:3],
        "disclaimer": "Bu bölüm kesin hüküm içermez; sadece pazarlık yaklaşımı ve belirsizlik yönetimi içindir."
    }


# =========================================================
# PERSONALIZED WARNINGS + CHECKLIST
# =========================================================
def build_personalized_warnings(enriched: Dict[str, Any], req: Optional[AnalyzeRequest] = None) -> List[str]:
    risk = enriched.get("risk", {}) or {}
    profile = enriched.get("profile", {}) or {}

    age = risk.get("age")
    mileage = risk.get("mileage_km") or 0
    usage = profile.get("usage", "mixed")
    info_q = enriched.get("info_quality", {}) or {}

    fuel = ""
    if req is not None:
        fuel = (req.vehicle.fuel or req.profile.fuel_preference or "") if hasattr(req, "vehicle") else ""
    if not fuel:
        fuel = profile.get("fuel_preference", "")

    warnings: List[str] = []

    if info_q.get("level") == "düşük":
        warnings.append("Bilgi seviyesi düşük: eksik alanlar var. Tahminler geniş band olarak verildi.")

    if age is not None and age <= 3 and mileage <= 60_000:
        warnings.append("Araç görece yeni ve düşük kilometreli; yine de garanti ve periyodik bakımları servis kayıtlarından doğrula.")
        warnings.append("Kaporta/boya ve iç mekân yıpranmasını detaylı kontrol et; fiyat pazarlığında kozmetiğin etkisi büyüktür.")
        if usage == "city":
            warnings.append("Şehir içi kullanımda fren/lastik ve süspansiyon darbelerini kontrol ettir.")
    elif (age is not None and age >= 10) or mileage >= 180_000:
        warnings.append("Yaş/km yükseldikçe motor, soğutma ve yürüyen aksam kontrolleri kritik hale gelir.")
        warnings.append("Otomatik şanzımanda test sürüşünde vuruntu/gecikme/titreme gibi belirtileri özellikle takip et.")
        warnings.append("Lift üzerinde alt takım ve direksiyon boşluklarını kontrol ettir; toplu değişimler ciddi maliyet çıkarabilir.")
    else:
        warnings.append("Yaş/km orta seviyede; hem mekanik hem kozmetik durumu birlikte değerlendir.")
        if usage == "city":
            warnings.append("Şehir içi ağırlıklı kullanımda fren ve süspansiyon elemanlarına daha dikkatli bak.")
        elif usage == "highway":
            warnings.append("Uzun yolda taş izleri, cam çatlakları ve yüksek hız balans titremelerini kontrol ettir.")

    f = str(fuel or "").lower()
    if "diesel" in f:
        warnings.append("Dizel motorda DPF/EGR ve enjektör sağlığını kontrol ettir; şehir içi kısa mesafede kurumlanma artabilir.")
    elif "lpg" in f:
        warnings.append("LPG'li araçta montaj kalitesi ve subap durumu kritik; kompresyon testi ve egzoz değerleri kontrol edilmeli.")
    elif "hybrid" in f or "electric" in f:
        warnings.append("Hibrit/EV araçlarda batarya sağlığı ve servis desteği uzun vadeli maliyeti belirler.")

    base_notes = risk.get("risk_notes") or []
    for r in base_notes:
        if r and r not in warnings:
            warnings.append(r)

    fuel_comment = risk.get("fuel_risk_comment")
    if fuel_comment and fuel_comment not in warnings:
        warnings.append(fuel_comment)

    return warnings[:8]


def build_buy_checklist(enriched: Dict[str, Any], req: Optional[AnalyzeRequest] = None, max_items: int = 6) -> List[str]:
    risk = enriched.get("risk", {}) or {}
    profile = enriched.get("profile", {}) or {}

    age = risk.get("age")
    mileage = risk.get("mileage_km") or 0

    fuel = ""
    if req is not None:
        fuel = (req.vehicle.fuel or req.profile.fuel_preference or "") if hasattr(req, "vehicle") else ""
    if not fuel:
        fuel = profile.get("fuel_preference", "")

    base_list = list(enriched.get("inspection_checklist") or [])
    checklist: List[str] = []

    if age is not None and age <= 3 and mileage <= 60_000:
        checklist.extend([
            "Kaporta/boya üzerinde parça değişimi, lokal boya ve ölçüm farklarını kontrol ettir.",
            "Servis kayıtlarını ve varsa garanti durumunu doğrula.",
            "İç mekân yıpranması kilometre ile uyumlu mu kontrol et.",
        ])
    elif (age is not None and age >= 10) or mileage >= 180_000:
        checklist.extend([
            "Motor için kompresyon/yağ kaçak kontrolü yaptır.",
            "Şanzımanda vuruntu/gecikme/titreme belirtilerini test sürüşünde kontrol et.",
            "Lift üzerinde alt takım/direksiyon boşluklarını kontrol ettir.",
        ])
    else:
        checklist.extend([
            "Rutin bakım kalemlerinin (yağ/filtre/fren) tarihini belgeyle doğrula.",
            "Lastik diş derinliği ve fren performansını test sürüşünde kontrol et.",
        ])

    f = str(fuel or "").lower()
    if "diesel" in f:
        checklist.append("Dizel: DPF/EGR durumu, enjektör kaçak testi ve turbo basıncını kontrol ettir.")
    elif "lpg" in f:
        checklist.append("LPG: sistemin ruhsata işli olması, montaj/proje belgeleri ve subap kontrolü.")

    for item in base_list:
        if len(checklist) >= max_items:
            break
        if item not in checklist:
            checklist.append(item)

    return checklist[:max_items]


# =========================================================
# PREMIUM TEMPLATE (deterministic) + "DOLU DOLU" SECTIONS
# =========================================================
def build_premium_template(req: AnalyzeRequest, enriched: Dict[str, Any]) -> Dict[str, Any]:
    v = req.vehicle
    title = f"{v.year or ''} {v.make} {v.model}".strip() or "Premium Analiz"

    costs = enriched["costs"]
    ins = enriched.get("insurance", {})
    taxes = enriched.get("taxes", {})
    fixed = enriched.get("fixed_costs", {})
    risk = enriched["risk"]
    prof = enriched["profile"]
    idx = (enriched.get("market", {}) or {}).get("indices", {}) or {}

    listed = costs.get("listed_price_try")
    maint_min, maint_max = costs["maintenance_yearly_try_min"], costs["maintenance_yearly_try_max"]
    maint_mid = int((maint_min + maint_max) / 2)
    fuel_mid = costs.get("yearly_fuel_tr_mid")

    maint_routine = int(costs.get("maintenance_routine_yearly_est") or int(maint_mid * 0.65))
    maint_reserve = int(costs.get("maintenance_risk_reserve_yearly_est") or max(0, maint_mid - maint_routine))
    total_mid = int(maint_mid + (fuel_mid or 0))

    cons_band = costs.get("consumption_l_per_100km_band") or (None, None)

    mtv = (taxes.get("mtv") or {})
    insp = (fixed.get("inspection") or {})

    traffic = (ins.get("traffic") or {})
    kasko = (ins.get("kasko") or {})

    parts_av = int(idx.get("parts_availability_score_1_5", 3))
    parts_cost = int(idx.get("parts_cost_index_1_5", 3))
    service_net = int(idx.get("service_network_index_1_5", 3))
    resale_liq = int(idx.get("resale_liquidity_score_1_5", 3))

    base_risk = risk.get("baseline_risk_level", "orta")
    overall = _score_from_risk(base_risk)

    uncertainty = build_uncertainty(enriched)
    flags = build_flags(enriched, req)
    indices_explain = explain_indices(enriched)
    value_block = build_value_and_negotiation(enriched)

    personal_fit_score = _fit_score(req, enriched)
    electronics_100 = _electronics_score(enriched)

    economy_100 = _clamp(78 - int(parts_cost * 2), 0, 100)
    comfort_100 = _clamp(72 if enriched["segment"]["code"] in ("D_SEDAN", "PREMIUM_D", "E_SEGMENT") else 66, 0, 100)
    family_use_100 = _clamp(74 if enriched["segment"]["code"] in ("C_SEDAN", "C_SUV", "D_SEDAN") else 66, 0, 100)
    resale_100 = _clamp(int(resale_liq * 18), 0, 100)

    # risk patterns (SAĞLAMLAŞTIRILDI: string gelirse patlamaz)
    risk_patterns = risk.get("risk_patterns") or []
    chronic_issues: List[str] = []
    for r in risk_patterns[:7]:
        if isinstance(r, dict):
            t = r.get("topic") or r.get("name") or "-"
            sev = r.get("severity") or r.get("level") or "orta"
            trig = r.get("trigger") or r.get("note") or "-"
            chronic_issues.append(f"{t} ({sev}) – tetik: {trig}")
        else:
            chronic_issues.append(str(r))

    warnings = build_personalized_warnings(enriched, req)
    checklist = build_buy_checklist(enriched, req, max_items=6)

    tek_bakis = [
        f"Skor: **{overall}/100** | Belirsizlik: **{uncertainty['level']}** (puan: {uncertainty['score_100']}/100)",
        f"Yıllık bakım bandı: **{_fmt_try(maint_min)} – {_fmt_try(maint_max)} TL** | Yakıt: **{_fmt_try(costs['yearly_fuel_tr_min'])} – {_fmt_try(costs['yearly_fuel_tr_max'])} TL**",
    ]
    if flags.get("red"):
        tek_bakis.append(f"En kritik nokta: {flags['red'][0]}")
    tek_bakis.append("Netleştirme: Tramer + servis kayıtları + OBD taraması skorun güvenini artırır.")

    # =========================================================
    # RESULT TEXT (premium ekranında gösterilecek)  ✅ DEĞİŞEN TEK YER BURASI
    # =========================================================
    lines: List[str] = []
    lines.append(f"## {title}")
    lines.append("")

    if (enriched.get("market", {}) or {}).get("profile_match") == "segment_estimate":
        lines.append("**Not:** Bu model için doğrudan veri profili bulunamadı; aynı segment emsallerinden **tahmini endeks/risk** üretildi.")
        lines.append("")

    iq = enriched.get("info_quality", {}) or {}
    missing_preview = ", ".join((iq.get("missing_fields", []) or [])[:6]) or "yok"
    lines.append(f"**Bilgi seviyesi:** {iq.get('level','-')} (eksikler: {missing_preview})")
    lines.append("")

    lines.append("---")
    lines.append("### 0) Tek bakış özeti")
    for t in tek_bakis[:4]:
        lines.append(f"- {t}")
    lines.append("---")
    lines.append("")

    lines.append("### 1) Oto Analiz Skoru (yön gösterici)")
    lines.append(f"- Genel: **{overall}/100**")
    lines.append(f"- Mekanik: **{_clamp(overall + 2, 0, 100)}/100** | Kaporta: **{_clamp(overall - 1, 0, 100)}/100**")
    lines.append(f"- Ekonomi: **{economy_100}/100** | Konfor: **{comfort_100}/100** | Aile: **{family_use_100}/100**")
    lines.append(f"- 2. el: **{resale_100}/100** | Elektronik: **{electronics_100}/100** | Uygunluk: **{personal_fit_score}/100**")
    lines.append("- Not: Skorlar **kesin teşhis** değildir; ilan+yaş+km+profil kombinasyonundan **tahmini** üretilir.")
    lines.append("")

    lines.append("### 2) Bayraklar (Kırmızı / Sarı / Yeşil)")
    if flags.get("red"):
        lines.append("- **Kırmızı (öncelikli kontrol):**")
        for x in flags["red"][:3]:
            lines.append(f"  - {x}")
    if flags.get("yellow"):
        lines.append("- **Sarı (belirsizlik / dikkat):**")
        for x in flags["yellow"][:3]:
            lines.append(f"  - {x}")
    if flags.get("green"):
        lines.append("- **Yeşil (artı taraflar):**")
        for x in flags["green"][:3]:
            lines.append(f"  - {x}")
    lines.append("")

    lines.append("---")
    lines.append("### 3) Yıllık maliyet özeti (tahmini band)")
    lines.append(f"- **Toplam (bakım + yakıt) orta:** **~{_fmt_try(total_mid)} TL/yıl**")
    lines.append(f"- Bakım bandı: **{_fmt_try(maint_min)} – {_fmt_try(maint_max)} TL/yıl**")
    lines.append(f"  - Rutin bakım tahmini: **~{_fmt_try(maint_routine)} TL**")
    lines.append(f"  - Risk payı (beklenmedik): **~{_fmt_try(maint_reserve)} TL**")
    lines.append(f"- Yakıt/enerji bandı: **{_fmt_try(costs['yearly_fuel_tr_min'])} – {_fmt_try(costs['yearly_fuel_tr_max'])} TL/yıl** (orta: {_fmt_try(fuel_mid)} TL)")
    if isinstance(cons_band, (list, tuple)) and len(cons_band) == 2 and cons_band[0] is not None:
        lines.append(f"- Tüketim bandı (segment tahmini): **{cons_band[0]} – {cons_band[1]} L/100km**")
    if mtv.get("ok"):
        lines.append(f"- MTV: **{_fmt_try(mtv['mtv_yearly_try_min'])} – {_fmt_try(mtv['mtv_yearly_try_max'])} TL/yıl**")
    if insp.get("ok"):
        lines.append(f"- Muayene yıllık ortalama: **~{_fmt_try(insp['annual_equivalent_try_2026_estimated'])} TL/yıl**")
    if traffic.get("ok"):
        mid = traffic.get("traffic_est_try_mid")
        mid_txt = f" (orta: {_fmt_try(mid)})" if isinstance(mid, int) else ""
        lines.append(f"- Trafik sigortası: **{_fmt_try(traffic['traffic_est_try_min'])} – {_fmt_try(traffic['traffic_est_try_max'])} TL/yıl**{mid_txt} (tavan: {_fmt_try(traffic['traffic_cap_try'])})")
    if kasko.get("ok"):
        kmid = kasko.get("kasko_try_mid")
        kmid_txt = f" (orta: {_fmt_try(kmid)})" if isinstance(kmid, int) else ""
        lines.append(f"- Kasko: **{_fmt_try(kasko['kasko_try_min'])} – {_fmt_try(kasko['kasko_try_max'])} TL/yıl**{kmid_txt}")
    lines.append("- Not: Bu bandlar, ilan bilgisi azaldıkça **daha geniş** tutulur.")
    lines.append("---")
    lines.append("")

    lines.append("### 4) Risk & kronik eğilimler (tahmini)")
    lines.append(f"- Baz risk seviyesi: **{base_risk}**")
    if chronic_issues:
        lines.append("- Öne çıkan risk paternleri:")
        for ci in chronic_issues[:5]:
            lines.append(f"  - {ci}")
    if warnings:
        lines.append("- Uyarılar:")
        for w in warnings[:6]:
            lines.append(f"  - {w}")
    lines.append("")

    lines.append("### 5) Parça / servis / 2. el endeksleri (puan + neye göre?)")
    lines.append(f"- Parça bulunabilirliği: **{parts_av}/5** → {indices_explain['parts_availability']['meaning']}")
    for w in indices_explain["parts_availability"]["why"][:2]:
        lines.append(f"  - Neye göre: {w}")
    lines.append(f"  - {indices_explain['parts_availability']['action']}")
    lines.append("")
    lines.append(f"- Parça maliyeti: **{parts_cost}/5** → {indices_explain['parts_cost']['meaning']}")
    for w in indices_explain["parts_cost"]["why"][:2]:
        lines.append(f"  - Neye göre: {w}")
    lines.append(f"  - {indices_explain['parts_cost']['action']}")
    lines.append("")
    lines.append(f"- Servis/usta ağı: **{service_net}/5** → {indices_explain['service_network']['meaning']}")
    for w in indices_explain["service_network"]["why"][:2]:
        lines.append(f"  - Neye göre: {w}")
    lines.append(f"  - {indices_explain['service_network']['action']}")
    lines.append("")
    lines.append(f"- 2. el likidite: **{resale_liq}/5** → {indices_explain['resale_liquidity']['meaning']}")
    for w in indices_explain["resale_liquidity"]["why"][:2]:
        lines.append(f"  - Neye göre: {w}")
    lines.append(f"  - {indices_explain['resale_liquidity']['action']}")
    lines.append("")

    lines.append("### 6) Kişiye uygunluk (profil bazlı)")
    lines.append(f"- Profil: yıllık **{prof['yearly_km']} km** ({prof['yearly_km_band']}), kullanım: **{prof['usage']}**, yakıt tercihi: **{prof['fuel_preference']}**")
    lines.append("- Bu bölüm; yıllık km ve kullanım tipine göre yakıt/segment mantığını yorumlar (kesin hüküm değil).")
    if prof["yearly_km_band"] == "düşük":
        lines.append("- Yıllık km düşükse, yüksek masraf potansiyelli seçeneklerde “gereksiz masraf” riski artabilir; bakım disiplini belirleyicidir.")
    if prof["yearly_km_band"] == "yüksek":
        lines.append("- Yıllık km yüksekse yakıt ekonomisi ve düzenli bakım en kritik iki değişkendir.")
    lines.append("")

    lines.append("### 7) Değer & pazarlık (hukuki güvenli, kesin hüküm yok)")
    if value_block.get("ok"):
        lines.append(f"- Fiyat etiketi: **{value_block.get('label')}**")
        lines.append(f"- Yorum: {value_block.get('comment')}")
        lines.append("- Pazarlık argümanları:")
        for a in (value_block.get("negotiation_args") or [])[:3]:
            lines.append(f"  - {a}")
        lines.append(f"- Not: {value_block.get('disclaimer')}")
    else:
        lines.append(f"- {value_block.get('note')}")
        for a in (value_block.get("negotiation_args") or [])[:2]:
            lines.append(f"  - {a}")
    lines.append("")

    lines.append("### 8) Belirsizlik & raporu netleştirme planı")
    lines.append(f"- Belirsizlik seviyesi: **{uncertainty['level']}** (puan: {uncertainty['score_100']}/100)")
    if uncertainty.get("how_to_improve"):
        lines.append("- Netleştirmek için:")
        for h in uncertainty["how_to_improve"][:4]:
            lines.append(f"  - {h}")
    lines.append("")

    lines.append("### 9) Satın almadan önce yapılacaklar (kısa checklist)")
    for c in checklist[:6]:
        lines.append(f"- {c}")
    lines.append("")

    # SON ÖZET (daha okunur, daha “final” hissi)
    lines.append("---")
    lines.append("### 10) Son özet (1 dakikalık karar notu)")
    if personal_fit_score >= 78:
        fit_label = "profilinle **uyumlu** görünüyor"
    elif personal_fit_score >= 62:
        fit_label = "profilinle **uyumlu ama dikkat isteyen** bir tablo var"
    else:
        fit_label = "profilinde **masraf/uyum riski daha yüksek** olabilir"

    lines.append(f"- Genel tablo: Skor **{overall}/100**, belirsizlik **{uncertainty['level']}** → bu araç {fit_label}.")
    lines.append(f"- Bütçe tarafı: Yıllık **bakım+yakıt orta ~{_fmt_try(total_mid)} TL** bandında düşün (sigorta/vergi ayrı).")
    if flags.get("red"):
        lines.append(f"- En kritik kontrol: **{flags['red'][0]}**")
    lines.append("- Kesin tespit için: **Ekspertiz + Tramer + OBD** birlikte değerlendirilmelidir (bu rapor kesin hüküm içermez).")
    lines.append("---")

    result_text = "\n".join(lines)

    # preview etiketi (fiyat varsa)
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
            "overall_100": overall,
            "mechanical_100": _clamp(overall + 2, 0, 100),
            "body_100": _clamp(overall - 1, 0, 100),
            "economy_100": economy_100,
            "comfort_100": comfort_100,
            "family_use_100": family_use_100,
            "resale_100": resale_100,
            "electronics_100": electronics_100,
            "personal_fit_100": personal_fit_score,
        },
        "cost_estimates": {
            "yearly_total_mid_try": int(maint_mid + (fuel_mid or 0)),
            "yearly_maintenance_tr": maint_mid,
            "yearly_maintenance_tr_min": maint_min,
            "yearly_maintenance_tr_max": maint_max,
            "yearly_fuel_tr": int(fuel_mid or 0),
            "yearly_fuel_tr_min": costs["yearly_fuel_tr_min"],
            "yearly_fuel_tr_max": costs["yearly_fuel_tr_max"],
            "notes": "Bandlar segment+yaş+km+profil ile tahmini üretilmiştir. Sigorta ve vergiler ayrı kalemlerdir.",
        },
        "flags": flags,
        "indices_explained": indices_explain,
        "uncertainty": uncertainty,
        "value_and_negotiation": value_block,
        "preview": {
            "title": title,
            "price_tag": price_tag,
            "spoiler": "Skorlar + bayraklar + maliyet bandı + kişiye uygunluk + belirsizlik planı",
            "bullets": [
                "Genel & alt skorlar",
                "Kırmızı/Sarı/Yeşil bayraklar",
                "Yıllık maliyet bandı",
                "Parça/servis/2.el endeksleri (neden?)",
                "Belirsizlik ve netleştirme planı",
            ],
        },
        "final_snapshot": {
            "tek_bakis_lines": tek_bakis[:4],
        },
        "result": result_text,
    }
    return out


# =========================================================
# LLM HELPERS (optional)
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
Sen Oto Analiz uygulaması için PREMIUM analiz metinlerini iyileştiren bir asistansın.
Senden sadece şu alanları daha doğal ve bilgili yapmanı istiyorum:
- summary.short_comment
- summary.pros (3-5 madde)
- summary.cons (3-5 madde)
- summary.who_should_buy (1-3 cümle)
- risk_analysis.warnings (5-10 madde, tekrar yok)

Kurallar:
- Rakamları UYDURMA, verilen sayıları değiştirme.
- 'alınır/alınmaz/sakın/tehlikeli' gibi kesin ifadeler kullanma.
- Dil Türkçe.
ÇIKTI SADECE JSON:
{
  "summary": {"short_comment":"", "pros":[], "cons":[], "who_should_buy":""},
  "risk_analysis": {"warnings":[]}
}
""".strip()


def premium_analyze_impl(req: AnalyzeRequest) -> Dict[str, Any]:
    enriched = build_enriched_context(req)
    base = build_premium_template(req, enriched)
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
- 'alınır/alınmaz/sakın/tehlikeli' kullanma.
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
            "pros": ["Ekspertiz ve tramer ile detaylar netleştirildiğinde daha sağlıklı yorum yapılır."],
            "cons": ["Bakım geçmişi ve ilan detayları netleşmeden skor bandı geniş kalabilir."],
            "estimated_risk_level": "orta",
        },
        "preview": {
            "title": title,
            "price_tag": None,
            "spoiler": "Genel değerlendirme hazır. Ekspertiz ve bakım geçmişi teyit edilmeden karar verilmemeli.",
            "bullets": ["Tramer/hasar kontrolü", "Bakım kayıtlarını sor", "Test sürüşü + OBD"],
        },
        "result": "Nötr bir ikinci el değerlendirmesi sağlandı. Detaylı karar için ilan açıklaması, ekspertiz raporu ve tramer sorgusu birlikte düşünülmelidir.",
    }


def call_llm_or_fallback(model_name: str, system_prompt: str, user_content: str, fallback_fn, req_obj):
    out = call_llm_json(model_name, system_prompt, user_content)
    if isinstance(out, dict):
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
        "hint": "Ekspertiz, tramer, bakım geçmişi ve kullanım profiline göre dengeli, tekrar etmeyen bir değerlendirme üret."
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
        "hint": "Bu araç kullanıcının kendi aracı olabilir; bakım planı ve riskleri daha pratik anlat."
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
        "summary": "İki araç için de ekspertiz ve tramer şart. Sol taraf varsayılan olarak daha dengeli kabul edildi.",
        "left_pros": ["Genel masraf/likidite dengesi daha öngörülebilir olabilir."],
        "left_cons": ["Kesin sonuç için ekspertiz + tramer gerekli."],
        "right_pros": ["Doğru bakım geçmişiyle mantıklı bir alternatif olabilir."],
        "right_cons": ["Masraf/risk tarafında daha dikkatli inceleme gerektirebilir."],
        "use_cases": {
            "family_use": "Aile kullanımı için sol taraf varsayımsal olarak daha avantajlı kabul edildi.",
            "long_distance": "Her iki araç da düzenli bakım ile uzun yolda kullanılabilir.",
            "city_use": "Şehir içinde şanzıman tipi ve tüketim belirleyici olur."
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
        "answer": "Bütçe, yıllık km ve kullanım tipine göre segment seçmek en mantıklısıdır. Ekspertiz + tramer şart.",
        "suggested_segments": ["C-sedan", "C-SUV", "B-Hatch"],
        "example_models": ["Toyota Corolla", "Renault Megane", "Honda Civic", "Hyundai Tucson"]
    }


if __name__ == "__main__":
    # Lokal çalıştırma için:
    # uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    pass
