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
    # TR: 1.234.567
    return f"{int(n):,}".replace(",", ".")


def _clamp(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, n))


def _parse_listed_price(req: AnalyzeRequest) -> Optional[int]:
    # Flutter context: listed_price_text
    try:
        txt = (req.context or {}).get("listed_price_text") or ""
        v = _digits(str(txt))
        if v and v > 10_000:
            return v
    except:
        pass

    # Bazı denemelerde "text" alanından da fiyat gelebilir
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
    """
    İlan metninden 1.6 / 2.0 / 1.5 gibi motor hacmi yakalama.
    1.6 -> 1600 cc
    """
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
    """
    Basit tag çıkarımı (metinden):
    - DSG/DCT/CVT
    - LPG
    - hybrid/electric
    - 4x4
    """
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

    # fuel üzerinden
    f = (req.vehicle.fuel or req.profile.fuel_preference or "").lower().strip()
    if f == "diesel":
        tags.append("diesel")
    if f == "lpg":
        tags.append("lpg_common")
    if f == "hybrid":
        tags.append("hybrid")
    if f == "electric":
        tags.append("ev")

    # uniq
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

    # hiç eşleşme yoksa segmentten emsal seç
    return [a for a in ANCHORS if a.get("segment") == segment][:limit]


def _profile_by_key(key: str) -> Optional[Dict[str, Any]]:
    nk = _norm(key)
    for p in VEHICLE_PROFILES:
        if _norm(p.get("key", "")) == nk:
            return p
    return None


def resolve_vehicle_profile(req: AnalyzeRequest, segment: str) -> Dict[str, Any]:
    """
    1) Anchor eşleşmesi varsa: o key üzerinden profile getir.
    2) Yoksa: make/model substring + aliases ile dene.
    3) Yoksa: segment ortalaması ile "tahmini" profile üret.
    """
    v = req.vehicle
    target = _norm(f"{v.make} {v.model}")
    anchors = find_anchor_matches(v.make, v.model, segment, limit=3)

    # 1) anchor üzerinden
    for a in anchors:
        p = _profile_by_key(a.get("key", ""))
        if p:
            return {
                "match": "exact_or_anchor",
                "matched_key": p.get("key"),
                "profile": p,
                "anchors_used": anchors,
            }

    # 2) substring + aliases
    for p in VEHICLE_PROFILES:
        k = _norm(p.get("key", ""))
        if k and k in target:
            return {"match": "substring", "matched_key": p.get("key"), "profile": p, "anchors_used": anchors}
        for al in (p.get("aliases") or []):
            if _norm(al) and _norm(al) in target:
                return {"match": "alias", "matched_key": p.get("key"), "profile": p, "anchors_used": anchors}

    # 3) segment average
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
    """
    MTV: 2026 tahmini tablo üzerinden hesaplar.
    - CC yoksa segmentten CC bandıyla min/max döner.
    - 2018+ değer bazlı tarifede araç değeri yoksa min/max band döner.
    """
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
    """
    Muayene: binek için (car_light) 2 yılda bir varsayımıyla yıllık ortalama verir.
    """
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
    """
    Trafik sigortası için resmi tavan (seed) bazlı tahmin bandı.
    - plaka_city_code: 34/06/35 gibi (context.plate_city_code)
    - traffic_step: 0..8 (context.traffic_step)
    """
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

    # Tipik band (tavanın altında). Hasarsızlık ve şirkete göre değişir.
    # Basamak yükseldikçe genelde daha aşağıda olur.
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
        "note": "Trafik sigortası için band, ilgili il ve basamak tavanına göre tahmini verilmiştir; gerçek teklif sürücü/hasar geçmişine göre değişir.",
    }


def estimate_kasko(req: AnalyzeRequest, segment_code: str, age: Optional[int], mileage_km: int) -> Dict[str, Any]:
    """
    Kasko: araç değerine göre oran bandı (segment + yaş + km ile ayarlanır).
    """
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

    # yaş/km katsayısı
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

    # çok uçuk bandları kırp
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
        "note": "Kasko bandı araç değeri + segment + yaş/km ile tahmini verilmiştir. Şehir, teminatlar, sürücü profili ve hasarsızlık indirimiyle ciddi değişir.",
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

    # Yakıt / enerji yıllık tahmini (TL bandı)
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
        risk_notes.append("Km çok yüksek; motor/şanzıman revizyon riski artmış olabilir.")
    elif mileage >= 180_000 and risk_level != "yüksek":
        risk_level = "orta-yüksek"
        risk_notes.append("Km yüksek; yürüyen aksam ve mekanik masraf ihtimali artmış olabilir.")

    if "yüksek" in fuel_risk:
        risk_level = "yüksek"

    if segment_code in ("PREMIUM_D", "E_SEGMENT") and ((age and age > 10) or mileage > 180_000):
        risk_notes.append("Premium sınıfta yaşlı/yüksek km araçların büyük masraf kalemleri çok pahalı olabilir.")

    # Trafik + Kasko
    traffic = estimate_traffic_insurance(req)
    kasko = estimate_kasko(req, segment_code=segment_code, age=age, mileage_km=mileage)

    # Vergiler + sabit giderler
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

    # tag riskleri: model profili varsa kullan; yoksa tag mapten üret
    guess_tags = _guess_tags(req)
    base_risks = (vp.get("risk_patterns") or [])
    base_watch = (vp.get("big_maintenance_watchlist") or [])

    # profile'da risk yoksa, tag map jsonundan üret
    if not base_risks:
        tag_map = _load_json(_dp("risk_patterns_by_tag_v1.json"), {})
        for t in guess_tags:
            base_risks.extend(tag_map.get(t, []))
    if not base_watch:
        watch_map = _load_json(_dp("big_maintenance_watchlist_by_tag_v1.json"), {})
        for t in guess_tags:
            base_watch.extend(watch_map.get(t, []))

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
            "resale_speed": None,
            "parts_availability": None,
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
# PREMIUM TEMPLATE (deterministic)
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

    mtv = (taxes.get("mtv") or {})
    insp = (fixed.get("inspection") or {})

    traffic = (ins.get("traffic") or {})
    kasko = (ins.get("kasko") or {})

    # Indeksler -> metin
    parts_av = idx.get("parts_availability_score_1_5", 3)
    parts_cost = idx.get("parts_cost_index_1_5", 3)
    service_net = idx.get("service_network_index_1_5", 3)
    resale_liq = idx.get("resale_liquidity_score_1_5", 3)

    # Risk seviyesi
    base_risk = risk.get("baseline_risk_level", "orta")
    overall = _score_from_risk(base_risk)

    # profile-based yorum
    personal_fit = [
        f"Kullanım profili: yıllık {prof['yearly_km']} km ({prof['yearly_km_band']}), kullanım tipi: {prof['usage']}.",
        "Şehir içi ağırlıklı kullanımda (özellikle dizel/otomatikte) DPF/şanzıman hassasiyetleri daha kritik olur.",
        "Uzun yol ağırlıklı kullanımda tüketim daha stabil olur; düzenli bakım kayıtları daha da değer kazanır.",
    ]
    if prof["yearly_km_band"] == "düşük":
        personal_fit.append("Yıllık km düşükse, dizel/premium masraflar gereksiz kalabilir; benzinli/hibrit gibi seçenekler daha mantıklı olabilir.")
    if prof["yearly_km_band"] == "yüksek":
        personal_fit.append("Yıllık km yüksekse, yakıt ekonomisi ve bakım disiplini en önemli iki belirleyici olur.")

    maintenance_breakdown = [
        f"Yıllık bakım bandı (tahmini): **{_fmt_try(maint_min)} – {_fmt_try(maint_max)} TL** (emsal segment + yaş/km + kullanım profiline göre).",
        f"Bu bandın içinde periyodik/öngörülebilir pay: yaklaşık **{_fmt_try(costs['maintenance_routine_yearly_est'])} TL/yıl**.",
        f"Beklenmedik masraflar için önerilen risk payı: yaklaşık **{_fmt_try(costs['maintenance_risk_reserve_yearly_est'])} TL/yıl**.",
    ]
    if risk.get("maintenance_watchlist"):
        maintenance_breakdown.append("Büyük bakım kapısı olabilecek kalemler: " + "; ".join([w.get("item","-") for w in risk["maintenance_watchlist"][:3]]))

    insurance_kasko = []
    if traffic.get("ok"):
        insurance_kasko.append(f"Trafik sigortası tahmini band: **{_fmt_try(traffic['traffic_est_try_min'])} – {_fmt_try(traffic['traffic_est_try_max'])} TL** (tavan: {_fmt_try(traffic['traffic_cap_try'])} TL).")
    else:
        insurance_kasko.append("Trafik sigortası: sürücü basamağı/il bilgisi eksik olduğu için net band üretilemedi.")
    if kasko.get("ok"):
        insurance_kasko.append(f"Kasko tahmini band: **{_fmt_try(kasko['kasko_try_min'])} – {_fmt_try(kasko['kasko_try_max'])} TL** (araç değerine göre tahmin).")
    else:
        insurance_kasko.append("Kasko: ilan fiyatı/araç değeri net değilse band geniş kalabilir (değer girilirse iyileşir).")

    taxes_block = []
    if mtv.get("ok"):
        taxes_block.append(f"MTV (tahmini {mtv.get('tax_year')}): **{_fmt_try(mtv['mtv_yearly_try_min'])} – {_fmt_try(mtv['mtv_yearly_try_max'])} TL/yıl** (taksit: ~{_fmt_try(mtv['mtv_installment_try_mid'])} TL).")
    else:
        taxes_block.append("MTV: motor hacmi / kayıt yılı / taşıt değeri bilgisi eksikse band üretmek zorlaşır.")
    if insp.get("ok"):
        taxes_block.append(f"Muayene (tahmini 2026): ücret ~{_fmt_try(insp['fee_try_2026_estimated'])} TL (binek 2 yılda bir varsayımıyla yıllık ortalama ~{_fmt_try(insp['annual_equivalent_try_2026_estimated'])} TL).")
    else:
        taxes_block.append("Muayene: veri bulunamadı.")

    parts_and_service = [
        f"Parça bulunabilirliği skoru: **{parts_av}/5** (1=zayıf, 5=çok iyi).",
        f"Parça maliyet endeksi: **{parts_cost}/5** (1=düşük, 5=yüksek).",
        f"Servis/usta ağı endeksi: **{service_net}/5** (1=zayıf, 5=çok yaygın).",
    ]

    resale_market = [
        f"İkinci el likidite skoru: **{resale_liq}/5** (1=yavaş, 5=hızlı).",
        "Temiz tramer + düzenli bakım kaydı + doğru paket/donanım satışı ciddi hızlandırır.",
        "Renk, otomatik/manuel, lastik-fren durumu ve ekspertiz raporu satış hızını belirgin etkiler.",
    ]

    negotiation_tips = [
        "Ekspertizde çıkan masrafları (lastik, fren, alt takım, bakım) tek tek kalemleyip pazarlık maddesi yap.",
        "Tramer toplamı ve parça değişimleri şasi/podye ile birlikte okunmalı; ağır hasar şüphesi varsa fiyatı agresif kırdır.",
        "Bakım kayıtları yoksa: yakın vadede yağ/buji/filtre + sıvılar + şanzıman bakımı için bütçe iste.",
    ]

    alternatives_same_segment = [
        "Aynı segmentte 2–3 farklı ilanla emsal kıyas yap (km + tramer + paket).",
        "Masrafı düşürmek için aynı bütçeyle daha düşük km’li/temiz geçmişli bir alternatif araştır.",
        "Yakıt tercihini kullanım profiline göre tekrar düşün: şehir içi + düşük km’de dizel gereksiz olabilir.",
    ]

    risk_patterns = risk.get("risk_patterns") or []
    chronic_issues = []
    for r in risk_patterns[:7]:
        t = r.get("topic") or "-"
        sev = r.get("severity") or "orta"
        trig = r.get("trigger") or "-"
        chronic_issues.append(f"{t} ({sev}) – tetik: {trig}")

    warnings = (risk.get("risk_notes") or [])[:]
    if risk.get("fuel_risk_comment"):
        warnings.append(risk["fuel_risk_comment"])
    if enriched.get("info_quality", {}).get("level") == "düşük":
        warnings.insert(0, "Bilgi seviyesi düşük: eksik alanlar var. Tahminler geniş band olarak verildi.")

    # Inspection checklist
    checklist = enriched.get("inspection_checklist") or []
    if checklist:
        negotiation_tips.append("Alım öncesi hızlı checklist: " + " | ".join(checklist[:5]))

    # Result (premium rapor)
    lines = []
    lines.append(f"## {title}")
    lines.append("")
    if enriched.get("market", {}).get("profile_match") == "segment_estimate":
        lines.append("**Not:** Bu model için doğrudan veri profili bulunamadı; aynı segmentteki popüler emsallerden tahmini endeks/risk üretildi.")
        lines.append("")
    iq = enriched.get("info_quality", {})
    lines.append(f"**Bilgi seviyesi:** {iq.get('level','-')} (eksikler: {', '.join(iq.get('missing_fields', [])[:6]) or 'yok'})")
    lines.append("")

    lines.append("### 1) Yıllık toplam maliyet özeti (tahmini band)")
    lines.append(f"- Bakım: **{_fmt_try(maint_min)} – {_fmt_try(maint_max)} TL/yıl**")
    lines.append(f"- Yakıt/enerji: **{_fmt_try(costs['yearly_fuel_tr_min'])} – {_fmt_try(costs['yearly_fuel_tr_max'])} TL/yıl** (orta: {_fmt_try(fuel_mid)} TL)")
    if mtv.get("ok"):
        lines.append(f"- MTV: **{_fmt_try(mtv['mtv_yearly_try_min'])} – {_fmt_try(mtv['mtv_yearly_try_max'])} TL/yıl**")
    if insp.get("ok"):
        lines.append(f"- Muayene yıllık ortalama: **~{_fmt_try(insp['annual_equivalent_try_2026_estimated'])} TL/yıl**")
    if traffic.get("ok"):
        lines.append(f"- Trafik sigortası: **{_fmt_try(traffic['traffic_est_try_min'])} – {_fmt_try(traffic['traffic_est_try_max'])} TL/yıl** (tavan: {_fmt_try(traffic['traffic_cap_try'])})")
    if kasko.get("ok"):
        lines.append(f"- Kasko: **{_fmt_try(kasko['kasko_try_min'])} – {_fmt_try(kasko['kasko_try_max'])} TL/yıl**")
    lines.append("")

    lines.append("### 2) Risk & kronik eğilimler (tahmini)")
    lines.append(f"- Baz risk seviyesi: **{base_risk}**")
    if chronic_issues:
        lines.append("- Öne çıkan risk paternleri:")
        for ci in chronic_issues[:5]:
            lines.append(f"  - {ci}")
    if warnings:
        lines.append("- Uyarılar:")
        for w in warnings[:5]:
            lines.append(f"  - {w}")
    lines.append("")

    lines.append("### 3) Parça/servis ve ikinci el")
    lines.append(f"- Parça bulunabilirliği: **{parts_av}/5** | Parça maliyet endeksi: **{parts_cost}/5**")
    lines.append(f"- Servis ağı: **{service_net}/5** | 2. el likidite: **{resale_liq}/5**")
    lines.append("")

    lines.append("### 4) Satın almadan önce yapılacaklar (kısa plan)")
    for c in checklist[:6]:
        lines.append(f"- {c}")
    lines.append("")

    lines.append("### 5) Pazarlık stratejisi")
    for t in negotiation_tips[:5]:
        lines.append(f"- {t}")
    lines.append("")

    lines.append("### 6) Son yorum")
    lines.append("Bu rapor; segment emsalleri, yaş/km ve kullanım profiline göre **band** üretir. Kesin karar için ekspertiz + tramer + OBD taraması mutlaka birlikte değerlendirilmeli.")

    result_text = "\n".join(lines)

    # preview: rakam yok
    price_tag = None
    if listed:
        # çok kaba etiket: bakım bandına göre değil, sadece fiyatın büyüklüğüne göre nötr
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
            "economy_100": _clamp(78 - int(parts_cost*2), 0, 100),
            "comfort_100": _clamp(72 if enriched["segment"]["code"] in ("D_SEDAN","PREMIUM_D","E_SEGMENT") else 66, 0, 100),
            "family_use_100": _clamp(74 if enriched["segment"]["code"] in ("C_SEDAN","C_SUV","D_SEDAN") else 66, 0, 100),
            "resale_100": _clamp(int(resale_liq*18), 0, 100),
        },
        "cost_estimates": {
            "yearly_maintenance_tr": maint_mid,
            "yearly_maintenance_tr_min": maint_min,
            "yearly_maintenance_tr_max": maint_max,
            "yearly_fuel_tr": int(fuel_mid or 0),
            "yearly_fuel_tr_min": costs["yearly_fuel_tr_min"],
            "yearly_fuel_tr_max": costs["yearly_fuel_tr_max"],
            "insurance_level": enriched["segment"].get("notes", ["orta"])[0] if enriched.get("segment") else "orta",
            "insurance_band_tr": None,
            "notes": "Bakım/yakıt bandları segment+yaş+km+profil ile tahmini üretilmiştir. Sigorta ve vergiler ayrı alanlarda verilir."
        },
        "risk_analysis": {
            "chronic_issues": chronic_issues[:6],
            "risk_level": base_risk,
            "warnings": warnings[:8],
        },
        "summary": {
            "short_comment": "Veriye dayalı bandlarla maliyet, risk ve ikinci el tarafını özetledim. Ekspertiz ve tramer doğrulamasıyla netleşir.",
            "pros": [
                "Masraf kalemleri band olarak ayrıştırıldı (bakım/yakıt/sigorta/vergiler).",
                "Parça/servis ve ikinci el likidite endeksleri eklendi.",
                "Ekspertiz checklist ve pazarlık planı ile aksiyona dönüştürüldü.",
            ],
            "cons": [
                "Tramer, bakım geçmişi ve şanzıman tipi netleşmeden risk bandı geniş kalabilir.",
                "Kasko/teklifler sürücü profiline göre ciddi değişir.",
                "İlan detayları kısa ise (paket/hasar/servis geçmişi) yorumlar daha genel kalır.",
            ],
            "who_should_buy": "Bütçesinde yıllık masraf bandını kabul eden, alım öncesi ekspertiz+OBD+trameri eksiksiz yapacak kullanıcılar için daha uygundur.",
        },
        "preview": {
            "title": title,
            "price_tag": price_tag,
            "spoiler": "Masraf bandı + risk paternleri + ikinci el/servis endeksleriyle premium rapor.",
            "bullets": [
                "Yıllık bakım & yakıt bandı",
                "Sigorta / MTV / muayene özeti",
                "Kronik risk + ekspertiz checklist",
                "Pazarlık stratejisi"
            ],
        },
        "details": {
            "personal_fit": personal_fit[:5],
            "maintenance_breakdown": maintenance_breakdown[:5],
            "insurance_kasko": insurance_kasko[:4],
            "parts_and_service": parts_and_service[:4],
            "resale_market": resale_market[:4],
            "negotiation_tips": negotiation_tips[:6],
            "alternatives_same_segment": alternatives_same_segment[:4],
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

    # LLM varsa, sadece metinleri iyileştir (bandları/hesapları değiştirmesin)
    rewrite_input = {
        "vehicle": req.vehicle.dict(),
        "profile": req.profile.dict(),
        "info_quality": enriched.get("info_quality"),
        "tags_detected": enriched.get("tags_detected"),
        "risk_patterns": (enriched.get("risk", {}) or {}).get("risk_patterns", [])[:6],
        "base_summary": base.get("summary"),
        "base_warnings": (base.get("risk_analysis") or {}).get("warnings", []),
    }
    llm = call_llm_json(
        model_name=OPENAI_MODEL_PREMIUM,
        system_prompt=SYSTEM_PROMPT_PREMIUM_REWRITE,
        user_content=json.dumps(rewrite_input, ensure_ascii=False),
    )
    if isinstance(llm, dict):
        try:
            if "summary" in llm and isinstance(llm["summary"], dict):
                base["summary"].update({k: llm["summary"][k] for k in ["short_comment", "pros", "cons", "who_should_buy"] if k in llm["summary"]})
            if "risk_analysis" in llm and isinstance(llm["risk_analysis"], dict) and "warnings" in llm["risk_analysis"]:
                base["risk_analysis"]["warnings"] = llm["risk_analysis"]["warnings"]
        except Exception:
            pass

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
- PREVIEW'da 'alınır/alınmaz/sakın/tehlikeli' kullanma.
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
            "pros": ["Ekspertiz ve tramer ile detaylar netleştirildiğinde mantıklı bir tercih olabilir."],
            "cons": ["Bakım geçmişi, km uyumu ve hasar kaydı netleşmeden kesin kanaat vermek doğru olmaz."],
            "estimated_risk_level": "orta",
        },
        "preview": {
            "title": title,
            "price_tag": None,
            "spoiler": "Genel değerlendirme hazır. Ekspertiz ve bakım geçmişi teyit edilmeden karar verilmemeli.",
            "bullets": ["Tramer/hasar kaydını kontrol et", "Bakım kayıtlarını sor", "Alt takım & lastik kontrolü"],
        },
        "result": "Genel, nötr bir ikinci el değerlendirmesi sağlandı. Detaylı karar için ilan açıklaması, ekspertiz raporu ve tramer sorgusu mutlaka birlikte düşünülmelidir.",
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
    # Normal analiz: LLM; yoksa fallback
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
    return call_llm_or_fallback(OPENAI_MODEL_NORMAL, SYSTEM_PROMPT_NORMAL, json.dumps(user_content, ensure_ascii=False), fallback_normal, req)


@app.post("/premium_analyze")
async def premium_analyze(req: AnalyzeRequest) -> Dict[str, Any]:
    # Premium: deterministic + optional LLM rewrite
    return premium_analyze_impl(req)


@app.post("/manual_analyze")
@app.post("/manual")
async def manual_analyze(req: AnalyzeRequest) -> Dict[str, Any]:
    # Manueli şimdilik normal analiz gibi ele al
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
    return call_llm_or_fallback(OPENAI_MODEL_NORMAL, SYSTEM_PROMPT_NORMAL, json.dumps(user_content, ensure_ascii=False), fallback_normal, req)


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
    # fallback
    return {
        "better_overall": "left",
        "summary": "İki araç için de ekspertiz ve tramer şart. Sol taraf varsayılan olarak daha dengeli kabul edildi.",
        "left_pros": ["Genel masraf/likidite dengesi daha öngörülebilir olabilir."],
        "left_cons": ["Kesin sonuç için ekspertiz + tramer gerekli."],
        "right_pros": ["Doğru bakım geçmişiyle mantıklı bir alternatif olabilir."],
        "right_cons": ["Masraf/risk tarafında daha dikkatli inceleme gerektirebilir."],
        "use_cases": {"family_use": "Aile kullanımı için sol taraf varsayımsal olarak daha avantajlı kabul edildi.",
                      "long_distance": "Her iki araç da düzenli bakım ile uzun yolda kullanılabilir.",
                      "city_use": "Şehir içinde şanzıman tipi ve tüketim belirleyici olur."}
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
