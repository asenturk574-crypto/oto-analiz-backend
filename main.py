import os, json, re, math
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


# =========================
# ENV + CLIENT
# =========================
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if (api_key and OpenAI is not None) else None

OPENAI_MODEL_DEFAULT = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_MODEL_NORMAL = os.getenv("OPENAI_MODEL_NORMAL", OPENAI_MODEL_DEFAULT)
OPENAI_MODEL_PREMIUM = os.getenv("OPENAI_MODEL_PREMIUM", OPENAI_MODEL_DEFAULT)
OPENAI_MODEL_COMPARE = os.getenv("OPENAI_MODEL_COMPARE", OPENAI_MODEL_DEFAULT)
OPENAI_MODEL_OTOBOT = os.getenv("OPENAI_MODEL_OTOBOT", OPENAI_MODEL_DEFAULT)

DATA_DIR = os.getenv("DATA_DIR", "data")
CURRENT_YEAR = int(os.getenv("CURRENT_YEAR", str(date.today().year)))


# =========================
# APP
# =========================
app = FastAPI(title="Oto Analiz Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


# =========================
# MODELS
# =========================
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


# =========================
# BASIC HELPERS
# =========================
def _dp(path_in_data: str) -> str:
    return os.path.join(DATA_DIR, path_in_data)

def _load_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return default

def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("ı","i").replace("ğ","g").replace("ş","s").replace("ö","o").replace("ü","u").replace("ç","c")
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

def _score10(x100: int) -> float:
    return round(_clamp(int(x100), 0, 100) / 10.0, 1)

def _price_tag(listed: Optional[int]) -> Optional[str]:
    if not listed:
        return None
    if listed < 500_000: return "Uygun"
    if listed < 1_200_000: return "Normal"
    return "Yüksek"

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
    if "dsg" in t or "s tronic" in t or "s-tronic" in t: tags.append("dsg")
    if "dct" in t or "edc" in t or "powershift" in t: tags.append("dct_optional")
    if "cvt" in t: tags.append("cvt")
    if "lpg" in t or "prins" in t or "landi" in t: tags.append("lpg_common")
    if "hybrid" in t: tags.append("hybrid")
    if "electric" in t or "ev" in t or "elektrik" in t: tags.append("ev")
    if "4x4" in t or "awd" in t or "quattro" in t: tags.append("awd_optional")
    if "turbo" in t or "tce" in t or "tsi" in t or "ecoboost" in t: tags.append("turbo_small")

    f = (req.vehicle.fuel or req.profile.fuel_preference or "").lower().strip()
    if f == "diesel": tags.append("diesel")
    if f == "lpg": tags.append("lpg_common")
    if f == "hybrid": tags.append("hybrid")
    if f == "electric": tags.append("ev")

    out: List[str] = []
    for x in tags:
        if x not in out:
            out.append(x)
    return out


# =========================
# DATA PACKS
# =========================
ANCHORS: List[Dict[str, Any]] = _load_json(_dp("anchors_tr_popular_96.json"), [])
VEHICLE_PROFILES: List[Dict[str, Any]] = _load_json(_dp("vehicle_profiles_96_v1.json"), [])
TRAFFIC_CAPS: Dict[str, Any] = _load_json(_dp("traffic_caps_tr_2025_12_seed.json"), {})
MTV_PACK: Dict[str, Any] = _load_json(_dp("mtv_tr_2025_2026_estimated_1895.json"), {})
FIXED_COSTS: Dict[str, Any] = _load_json(_dp("fixed_costs_tr_2026_estimated.json"), {})

TAG_RISK_MAP: Dict[str, Any] = _load_json(_dp("risk_patterns_by_tag_v1.json"), {})
WATCH_MAP: Dict[str, Any] = _load_json(_dp("big_maintenance_watchlist_by_tag_v1.json"), {})
CHECKLISTS_BY_SEG: Dict[str, Any] = _load_json(_dp("inspection_checklists_by_segment_v1.json"), {})


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


SEGMENT_PROFILES: Dict[str, Dict[str, Any]] = {
    "B_HATCH": {"name": "B segment (küçük hatchback)", "maintenance_yearly_range": (12000, 25000), "insurance_level": "orta",
        "notes": ["Şehir içi için pratik; parça/usta erişimi genelde rahattır.", "Bakım disipliniyle yıllık masraf çoğu zaman kontrol edilebilir kalır."]},
    "C_SEDAN": {"name": "C segment (aile sedan/hatchback)", "maintenance_yearly_range": (15000, 32000), "insurance_level": "orta",
        "notes": ["Türkiye’de likiditesi yüksek segmentlerden; temiz örneklerin alıcısı olur.", "Konfor/donanım artarken bakım maliyeti B segmente göre biraz yükselir."]},
    "C_SUV": {"name": "C segment SUV", "maintenance_yearly_range": (18000, 38000), "insurance_level": "orta-yüksek",
        "notes": ["Lastik/fren/alt takım masrafı gövde/ağırlık nedeniyle artabilir.", "Aile/uzun yol için dengeli; tüketim ve lastik maliyeti biraz daha yüksek olabilir."]},
    "D_SEDAN": {"name": "D segment (konfor sedan)", "maintenance_yearly_range": (22000, 50000), "insurance_level": "orta-yüksek",
        "notes": ["Donanım/konfor artışıyla parça ve işçilik maliyetleri belirgin yükselir."]},
    "PREMIUM_D": {"name": "Premium sınıf", "maintenance_yearly_range": (32000, 75000), "insurance_level": "yüksek",
        "notes": ["Premium sınıfta işçilik/parça maliyeti yüksektir; yaş/km yükseldikçe elektronik kalemler önem kazanır."]},
    "E_SEGMENT": {"name": "Üst sınıf", "maintenance_yearly_range": (45000, 120000), "insurance_level": "çok yüksek",
        "notes": ["Büyük gövde + yüksek donanım = masraf kalemleri bariz şekilde artar."]},
}


# =========================
# SEGMENT DETECTION
# =========================
def detect_segment(make: str, model: str) -> str:
    s = _norm(f"{make} {model}")
    if any(k in s for k in ["bmw", "mercedes", "audi", "volvo", "lexus", "range rover", "land rover"]):
        return "PREMIUM_D"
    if any(k in s for k in ["clio","polo","i20","corsa","yaris","fiesta","fabia","ibiza"]):
        return "B_HATCH"
    if any(k in s for k in ["corolla","civic","megane","astra","focus","egea","tipo","elantra","i30"]):
        return "C_SEDAN"
    if any(k in s for k in ["qashqai","tucson","sportage","kuga","3008","duster","qq"]):
        return "C_SUV"
    if any(k in s for k in ["passat","superb","508","insignia","camry"]):
        return "D_SEDAN"
    return "C_SEDAN"


def find_anchor_matches(make: str, model: str, segment: str, limit: int = 3) -> List[Dict[str, Any]]:
    target = _norm(f"{make} {model}")
    scored: List[Tuple[int, Dict[str, Any]]] = []
    for a in ANCHORS:
        score = 0
        if a.get("segment") == segment: score += 5
        key = _norm(a.get("key", ""))
        if key and key in target: score += 10
        for al in (a.get("aliases") or []):
            if _norm(al) in target: score += 8
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
            return {"match": "exact_or_anchor", "matched_key": p.get("key"), "profile": p, "anchors_used": anchors}

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


# =========================
# INFO QUALITY
# =========================
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
    ad_len = len(ad_text)
    if ad_len < 40: missing.append("ilan_aciklamasi_kisa")

    if p.usage not in ("city","mixed","highway"): missing.append("profil_kullanim_tipi")

    if len(missing) <= 1 and ad_len >= 120: level = "yüksek"
    elif len(missing) <= 3 and ad_len >= 40: level = "orta"
    else: level = "düşük"

    # belirsizlik: yüksek->düşük
    if level == "yüksek": uncertainty = "düşük"
    elif level == "orta": uncertainty = "orta"
    else: uncertainty = "yüksek"

    return {"level": level, "missing_fields": missing, "ad_length": ad_len, "uncertainty": uncertainty}


# =========================
# TAXES + FIXED COSTS
# =========================
def _age_group_car(age: int) -> str:
    if age <= 3: return "1_3"
    if age <= 6: return "4_6"
    if age <= 11: return "7_11"
    if age <= 15: return "12_15"
    return "16_plus"

def estimate_mtv(req: AnalyzeRequest, tax_year: Optional[int] = None) -> Dict[str, Any]:
    pack = MTV_PACK or {}
    if not pack: return {"ok": False, "note": "MTV pack bulunamadı."}

    table_container = pack.get("mtv_2026_estimated") or pack.get("mtv_2025_official")
    if not table_container: return {"ok": False, "note": "MTV tablo alanı eksik."}

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
    if reg_year: age = max(0, ty - reg_year)
    elif v.year: age = max(0, ty - v.year)

    age_key = _age_group_car(age)
    is_value_based = (reg_year is not None and reg_year >= 2018)

    table = table_container.get("tariff_I_automobile_value_based") if is_value_based else table_container.get("tariff_IA_pre2018_automobile")
    table = table or []
    segment = detect_segment(v.make, v.model)

    if engine_cc is None:
        seg_cc_band = {
            "B_HATCH": (0, 1600), "C_SEDAN": (0, 1600), "C_SUV": (1301, 2000),
            "D_SEDAN": (1601, 2000), "PREMIUM_D": (1601, 3000), "E_SEGMENT": (2001, 99999),
        }.get(segment, (0, 2000))
        cc_candidates = [seg_cc_band[0], seg_cc_band[1]]
        cc_note = "Motor hacmi bulunamadı; segment bazlı cc bandından tahmini MTV üretildi."
    else:
        cc_candidates = [engine_cc]
        cc_note = None

    candidates: List[Tuple[int, int, str]] = []
    for cc in cc_candidates:
        rows = [r for r in table if (r.get("cc_min", 0) <= cc <= r.get("cc_max", 999999))]
        if not rows: continue

        if is_value_based:
            if vehicle_value is None:
                vals = [int(r["tax"][age_key]) for r in rows if r.get("tax") and age_key in r["tax"]]
                if vals: candidates.append((min(vals), max(vals), "I/value_unknown"))
            else:
                def in_value(r: Dict[str, Any]) -> bool:
                    vmin = r.get("value_min", 0) or 0
                    vmax = r.get("value_max", None)
                    okmin = (vehicle_value > vmin) if r.get("value_min_exclusive") else (vehicle_value >= vmin)
                    okmax = True if vmax is None else (vehicle_value <= vmax)
                    return okmin and okmax
                rr = [r for r in rows if in_value(r)]
                if rr and rr[0].get("tax") and age_key in rr[0]["tax"]:
                    val = int(rr[0]["tax"][age_key])
                    candidates.append((val, val, "I/exact"))
                else:
                    vals = [int(r["tax"][age_key]) for r in rows if r.get("tax") and age_key in r["tax"]]
                    if vals: candidates.append((min(vals), max(vals), "I/value_fallback"))
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
    if cc_note: notes.append(cc_note)
    if is_value_based and vehicle_value is None:
        notes.append("2018+ değer bazlı tarifede araç değeri bulunamadığı için MTV band olarak verildi.")

    return {
        "ok": True, "tax_year": ty, "age": age,
        "engine_cc_used": engine_cc, "vehicle_value_used_try": vehicle_value,
        "tariff_basis": sorted(set(c[2] for c in candidates)),
        "mtv_yearly_try_min": mtv_min, "mtv_yearly_try_max": mtv_max, "mtv_yearly_try_mid": mtv_mid,
        "mtv_installment_try_min": int(mtv_min/2), "mtv_installment_try_max": int(mtv_max/2), "mtv_installment_try_mid": int(mtv_mid/2),
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


# =========================
# INSURANCE + COST ESTIMATION
# =========================
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
    mult_min = 0.45 if step in ("7","8") else 0.50
    mult_mid = 0.65 if step in ("7","8") else 0.70
    return {
        "ok": True,
        "city_code": city_code,
        "city_name": cities.get(city_code, {}).get("name"),
        "traffic_step": step,
        "traffic_cap_try": cap,
        "traffic_est_try_min": int(cap*mult_min),
        "traffic_est_try_mid": int(cap*mult_mid),
        "traffic_est_try_max": int(cap),
        "note": "Band tahminidir; sürücü geçmişi ve şirket politikasına göre değişir.",
    }

def estimate_kasko(req: AnalyzeRequest, segment_code: str, age: Optional[int], mileage_km: int) -> Dict[str, Any]:
    listed_price = _parse_listed_price(req)
    if not listed_price or listed_price < 100_000:
        return {"ok": False, "note": "Kasko için araç değeri bulunamadı (fiyat yok veya düşük)."}
    base = {
        "B_HATCH": (0.020, 0.045), "C_SEDAN": (0.022, 0.050), "C_SUV": (0.025, 0.055),
        "D_SEDAN": (0.028, 0.065), "PREMIUM_D": (0.030, 0.080), "E_SEGMENT": (0.035, 0.095),
    }.get(segment_code, (0.022, 0.055))

    age_mult = 1.0
    if age is not None:
        if age >= 15: age_mult = 1.35
        elif age >= 10: age_mult = 1.22
        elif age >= 6: age_mult = 1.10

    km_mult = 1.0
    if mileage_km >= 250_000: km_mult = 1.25
    elif mileage_km >= 180_000: km_mult = 1.15
    elif mileage_km >= 120_000: km_mult = 1.08

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
        "note": "Band tahminidir; il/teminatlar/hasarsızlık ve sürücü profiline göre ciddi değişir.",
    }

def estimate_costs(req: AnalyzeRequest) -> Dict[str, Any]:
    v = req.vehicle
    p = req.profile or Profile()

    segment_code = detect_segment(v.make, v.model)
    seg = SEGMENT_PROFILES.get(segment_code, SEGMENT_PROFILES["C_SEDAN"])
    listed_price = _parse_listed_price(req)

    age = max(0, CURRENT_YEAR - v.year) if v.year else None
    mileage = int(v.mileage_km or 0)

    base_min, base_max = seg["maintenance_yearly_range"]

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
    fuel_hint = "orta"
    if fuel == "diesel":
        fuel_mult_maint = 1.05
        if p.usage == "city" and mileage >= 120_000:
            fuel_hint = "orta-yüksek (şehir içi + yüksek km’de DPF/EGR tarafı daha sık takip ister)"
    elif fuel == "lpg":
        fuel_mult_maint = 0.95
        fuel_hint = "orta (montaj/ayar kalitesi belirleyici)"
    elif fuel in ("hybrid","electric"):
        fuel_mult_maint = 0.9
        fuel_hint = "düşük-orta (batarya sağlığına bağlı)"

    maint_min = int(base_min * age_mult * km_mult * usage_mult * fuel_mult_maint)
    maint_max = int(base_max * age_mult * km_mult * usage_mult * fuel_mult_maint)

    if listed_price and listed_price > 100_000:
        ratio_map = {
            "B_HATCH": (0.015, 0.06), "C_SEDAN": (0.015, 0.07), "C_SUV": (0.02, 0.08),
            "D_SEDAN": (0.02, 0.085), "PREMIUM_D": (0.025, 0.09), "E_SEGMENT": (0.03, 0.10),
        }
        r_min, r_max = ratio_map.get(segment_code, (0.015, 0.07))
        hard_min = int(listed_price * r_min)
        hard_max = int(listed_price * r_max)
        maint_min = max(maint_min, hard_min)
        maint_max = min(maint_max, hard_max)
        caps = {"B_HATCH": 45_000, "C_SEDAN": 55_000, "C_SUV": 75_000, "D_SEDAN": 85_000, "PREMIUM_D": 130_000, "E_SEGMENT": 160_000}
        maint_max = min(maint_max, caps.get(segment_code, 60_000))
        if maint_min > maint_max:
            maint_min = int(maint_max * 0.7)

    mid_maint = int((maint_min + maint_max) / 2)
    routine_est = int(mid_maint * 0.65)
    risk_reserve_est = max(0, mid_maint - routine_est)

    km_year = max(0, int(p.yearly_km or 15_000))
    km_factor = max(0.5, min(2.5, km_year / 15_000.0))

    fuel_base = {
        "B_HATCH": (18_000, 32_000), "C_SEDAN": (22_000, 40_000), "C_SUV": (26_000, 48_000),
        "D_SEDAN": (27_000, 52_000), "PREMIUM_D": (32_000, 65_000), "E_SEGMENT": (38_000, 80_000),
    }.get(segment_code, (22_000, 42_000))

    fuel_mult_cost = 1.0
    if fuel == "diesel": fuel_mult_cost = 0.9
    elif fuel == "lpg": fuel_mult_cost = 0.75
    elif fuel in ("hybrid","electric"): fuel_mult_cost = 0.7

    fuel_min = int(fuel_base[0] * km_factor * fuel_mult_cost)
    fuel_max = int(fuel_base[1] * km_factor * fuel_mult_cost)
    fuel_mid = int((fuel_min + fuel_max) / 2)

    risk_level = "orta"
    risk_notes: List[str] = []
    if age is not None and age >= 15:
        risk_level = "yüksek"; risk_notes.append("Araç yaşı yüksek; büyük bakım ihtimali artabilir.")
    elif age is not None and age >= 10:
        risk_level = "orta-yüksek"; risk_notes.append("Yaş nedeniyle büyük bakım kalemi çıkma ihtimali artar.")
    if mileage >= 250_000:
        risk_level = "yüksek"; risk_notes.append("Km çok yüksek; motor/şanzıman tarafı daha yakından izlenmeli.")
    elif mileage >= 180_000 and risk_level != "yüksek":
        risk_level = "orta-yüksek"; risk_notes.append("Km yüksek; yürüyen aksam ve mekanik kalemler öne çıkar.")

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
        "fuel_risk_comment": fuel_hint,
        "risk_level": risk_level,
        "risk_notes": risk_notes,
        "segment_notes": seg.get("notes", []),
        "consumption_l_per_100km_band": {
            "B_HATCH": (5.5, 7.5), "C_SEDAN": (6.5, 9.0), "C_SUV": (7.5, 11.0),
            "D_SEDAN": (7.5, 11.0), "PREMIUM_D": (8.0, 12.5), "E_SEGMENT": (9.0, 15.0),
        }.get(segment_code, (6.5, 9.5)),
        "taxes": {"mtv": mtv},
        "fixed_costs": {"inspection": inspection},
    }


# =========================
# ENRICHED CONTEXT
# =========================
def build_enriched_context(req: AnalyzeRequest) -> Dict[str, Any]:
    v = req.vehicle
    p = req.profile or Profile()

    seg_info = estimate_costs(req)
    segment_code = seg_info["segment_code"]

    anchors = find_anchor_matches(v.make, v.model, segment_code, limit=3)
    info_q = evaluate_info_quality(req)

    yearly_km_band = "orta"
    if p.yearly_km <= 8000: yearly_km_band = "düşük"
    elif p.yearly_km >= 30000: yearly_km_band = "yüksek"

    prof = resolve_vehicle_profile(req, segment_code)
    vp = prof["profile"] or {}

    guess_tags = _guess_tags(req)
    base_risks = (vp.get("risk_patterns") or [])
    base_watch = (vp.get("big_maintenance_watchlist") or [])

    if not base_risks:
        for t in guess_tags:
            base_risks.extend(TAG_RISK_MAP.get(t, []))
    if not base_watch:
        for t in guess_tags:
            base_watch.extend(WATCH_MAP.get(t, []))

    checklist = vp.get("inspection_checklist") or []
    if not checklist:
        checklist = (CHECKLISTS_BY_SEG.get(segment_code, []) or [])

    return {
        "segment": {"code": segment_code, "name": seg_info["segment_name"], "notes": seg_info.get("segment_notes", [])},
        "market": {"insurance_level": seg_info["insurance_level"], "indices": vp.get("indices") or {}, "profile_match": prof.get("match"), "matched_key": prof.get("matched_key")},
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
            "risk_patterns": (base_risks or [])[:12],
            "maintenance_watchlist": (base_watch or [])[:12],
        },
        "profile": {"yearly_km": p.yearly_km, "yearly_km_band": yearly_km_band, "usage": p.usage, "fuel_preference": p.fuel_preference},
        "info_quality": info_q,
        "anchors_used": [{"key": a.get("key"), "segment": a.get("segment"), "tags": a.get("tags", [])} for a in anchors],
        "inspection_checklist": (checklist or [])[:10],
        "tags_detected": guess_tags,
    }


# =========================
# PREMIUM SCORING (deterministic, "dolu dolu")
# =========================
def _risk_to_base_score(risk_level: str) -> int:
    # daha yumuşak mapping (çok “dip” olmaması için)
    if risk_level == "düşük": return 82
    if risk_level == "orta": return 76
    if risk_level == "orta-yüksek": return 69
    if risk_level == "yüksek": return 60
    return 74

def compute_personal_fit(enriched: Dict[str, Any], req: AnalyzeRequest) -> Dict[str, Any]:
    prof = enriched.get("profile", {}) or {}
    seg = enriched.get("segment", {}) or {}
    risk = enriched.get("risk", {}) or {}

    yearly_km = int(prof.get("yearly_km") or 15000)
    usage = str(prof.get("usage") or "mixed")
    fuel_pref = str(prof.get("fuel_preference") or "").lower()
    fuel = str(req.vehicle.fuel or fuel_pref or "gasoline").lower()
    segment_code = str(seg.get("code") or "C_SEDAN")
    age = risk.get("age")
    km = int(risk.get("mileage_km") or 0)

    score = 75
    reasons: List[str] = []

    # Yıllık km bandı
    if yearly_km <= 8000:
        score -= 6
        reasons.append("Yıllık km düşük: sabit giderlerin (sigorta/vergiler) km başı etkisi artar.")
    elif yearly_km >= 30000:
        score += 4
        reasons.append("Yıllık km yüksek: tüketim/konfor ve bakım disiplini daha belirleyici olur (profil net).")
    else:
        reasons.append("Yıllık km orta: maliyet ve kullanım dengesi açısından nötr bir profil.")

    # Kullanım tipi
    if usage == "city":
        score -= 4
        reasons.append("Şehir içi ağırlık: fren/lastik/alt takım gibi kalemler daha sık takip ister.")
    elif usage == "highway":
        score += 2
        reasons.append("Uzun yol ağırlık: tüketim daha stabil olabilir; düzenli bakım kaydı değer kazanır.")
    else:
        reasons.append("Karma kullanım: hem şehir içi hem uzun yol senaryosu için dengeli değerlendirilir.")

    # Yakıt tercih uyumu
    if fuel == "diesel":
        if yearly_km >= 20000 and usage in ("mixed","highway"):
            score += 5
            reasons.append("Dizel + yüksek km/uzun yol: kullanım mantığıyla daha uyumlu.")
        elif usage == "city" and yearly_km <= 12000:
            score -= 10
            reasons.append("Dizel + şehir içi + düşük km: DPF/EGR gibi takip ihtiyacı km’ye göre daha “yoğun” hissedilebilir.")
        else:
            reasons.append("Dizel: kullanım alışkanlığına göre takip gereksinimi değişebilir.")
    elif fuel == "lpg":
        score += 2
        reasons.append("LPG: yakıt maliyetinde avantaj sağlar; montaj/ayar kalitesi ve subap takibi belirleyicidir.")
    elif fuel in ("hybrid","electric"):
        score += 3
        reasons.append("Hibrit/EV: şehir içi verimlilik avantajı olabilir; batarya sağlığı/servis altyapısı kritik.")
    else:
        reasons.append("Benzinli: şehir içi kullanımda genelde daha sorunsuz ve esnek bir kullanım sunar.")

    # Segment pratikliği (profil)
    if segment_code in ("PREMIUM_D","E_SEGMENT"):
        score -= 4
        reasons.append("Premium/üst sınıf: parça/işçilik ve sigorta kalemleri profil bütçesinde daha belirgin yer kaplayabilir.")
    elif segment_code in ("C_SUV",):
        reasons.append("SUV gövde: lastik/fren/alt takım kalemleri sedan/hatch’e göre biraz daha öne çıkabilir.")
    else:
        reasons.append("Segment pratikliği: Türkiye koşullarında genel kullanıcı profiline yakın.")

    # Yaş/km ile “profil uyumu”
    if (age is not None and age >= 10) or km >= 180_000:
        score -= 4
        reasons.append("Yaş/km yüksekse, kullanım tarzından bağımsız olarak bakım planı daha disiplinli olmalı.")
    else:
        reasons.append("Yaş/km aşırı uçta değilse, profil uyumu daha rahat yakalanır.")

    return {"personal_fit_100": _clamp(int(score), 0, 100), "reasons": reasons[:6]}

def compute_subscores(enriched: Dict[str, Any], req: AnalyzeRequest) -> Dict[str, Any]:
    risk = enriched.get("risk", {}) or {}
    market = enriched.get("market", {}) or {}
    costs = enriched.get("costs", {}) or {}
    segment_code = str((enriched.get("segment", {}) or {}).get("code") or "C_SEDAN")

    base_risk = str(risk.get("baseline_risk_level") or "orta")
    age = risk.get("age")
    km = int(risk.get("mileage_km") or 0)
    fuel = str(req.vehicle.fuel or req.profile.fuel_preference or "").lower()
    usage = str((enriched.get("profile", {}) or {}).get("usage") or "mixed")

    idx = (market.get("indices") or {})
    parts_cost = int(idx.get("parts_cost_index_1_5", 3))
    service_net = int(idx.get("service_network_index_1_5", 3))
    resale_liq = int(idx.get("resale_liquidity_score_1_5", 3))

    base = _risk_to_base_score(base_risk)

    # Mekanik
    mech = base
    if age is not None:
        if age >= 15: mech -= 14
        elif age >= 10: mech -= 10
        elif age >= 6: mech -= 6
        elif age <= 3: mech += 2
    if km >= 250_000: mech -= 14
    elif km >= 180_000: mech -= 10
    elif km >= 120_000: mech -= 6
    elif km <= 60_000 and (age is not None and age <= 6): mech += 2
    if fuel == "diesel" and usage == "city" and km >= 120_000: mech -= 5
    if segment_code in ("PREMIUM_D","E_SEGMENT"): mech -= 2
    mech = _clamp(int(mech), 0, 100)

    # Elektronik/Donanım (premium ve yaşla oynar)
    elec = base + 2
    if segment_code in ("PREMIUM_D","E_SEGMENT"): elec += 2
    if age is not None and age >= 10: elec -= 8
    elif age is not None and age >= 6: elec -= 4
    if km >= 180_000: elec -= 4
    elec = _clamp(int(elec), 0, 100)

    # Maliyet Dengesi: parça maliyeti + yıllık toplam / fiyat oranı
    listed = costs.get("listed_price_try")
    maint_mid = int((int(costs.get("maintenance_yearly_try_min") or 0) + int(costs.get("maintenance_yearly_try_max") or 0)) / 2) if costs.get("maintenance_yearly_try_max") else int(costs.get("maintenance_yearly_try_min") or 0)
    fuel_mid = int(costs.get("yearly_fuel_tr_mid") or 0)

    traffic = ((enriched.get("insurance", {}) or {}).get("traffic") or {})
    kasko = ((enriched.get("insurance", {}) or {}).get("kasko") or {})
    mtv = (((enriched.get("taxes", {}) or {}).get("mtv") or {}))
    insp = (((enriched.get("fixed_costs", {}) or {}).get("inspection") or {}))

    traffic_mid = int(traffic.get("traffic_est_try_mid") or 0) if traffic.get("ok") else 0
    kasko_mid = int(kasko.get("kasko_try_mid") or 0) if kasko.get("ok") else 0
    mtv_mid = int(mtv.get("mtv_yearly_try_mid") or 0) if mtv.get("ok") else 0
    insp_mid = int(insp.get("annual_equivalent_try_2026_estimated") or 0) if insp.get("ok") else 0

    annual_mid = maint_mid + fuel_mid + traffic_mid + kasko_mid + mtv_mid + insp_mid

    cost_score = 75
    cost_score -= (parts_cost - 3) * 6  # 1..5 -> etkili
    cost_score += (service_net - 3) * 2

    if isinstance(listed, int) and listed > 0:
        ratio = annual_mid / float(listed)  # yıllık / araç değeri
        # ratio küçükse iyi
        if ratio <= 0.04: cost_score += 8
        elif ratio <= 0.06: cost_score += 3
        elif ratio <= 0.08: cost_score -= 4
        elif ratio <= 0.10: cost_score -= 10
        else: cost_score -= 16
    else:
        # fiyat yoksa segment bazlı biraz daha genel
        if segment_code in ("PREMIUM_D","E_SEGMENT"): cost_score -= 8
        elif segment_code in ("C_SUV","D_SEDAN"): cost_score -= 4

    cost_score = _clamp(int(cost_score), 0, 100)

    # 2. El Likidite
    resale = _clamp(int(45 + resale_liq * 11), 0, 100)
    if (age is not None and age >= 10) or km >= 180_000:
        resale -= 4
    resale = _clamp(int(resale), 0, 100)

    # Kişiye uygunluk
    pfit = compute_personal_fit(enriched, req)
    personal_fit_100 = int(pfit["personal_fit_100"])

    # Overall (ağırlıklı)
    overall = int(
        mech * 0.30 +
        elec * 0.20 +
        cost_score * 0.25 +
        resale * 0.15 +
        personal_fit_100 * 0.10
    )
    overall = _clamp(overall, 0, 100)

    return {
        "overall_100": overall,
        "mechanical_100": mech,
        "electronics_100": elec,
        "economy_100": cost_score,
        "resale_100": resale,
        "personal_fit_100": personal_fit_100,
        "annual_mid_try": annual_mid,
        "personal_fit_reasons": pfit.get("reasons") or [],
    }


# =========================
# PREMIUM TEMPLATE (Markdown result + JSON)
# =========================
def build_buy_checklist(enriched: Dict[str, Any], req: AnalyzeRequest, max_items: int = 6) -> List[str]:
    risk = enriched.get("risk", {}) or {}
    prof = enriched.get("profile", {}) or {}
    age = risk.get("age")
    km = int(risk.get("mileage_km") or 0)
    usage = str(prof.get("usage") or "mixed")
    fuel = str(req.vehicle.fuel or req.profile.fuel_preference or "").lower()

    base_list = list(enriched.get("inspection_checklist") or [])
    out: List[str] = []

    if age is not None and age <= 3 and km <= 60_000:
        out += [
            "Kaporta/boya kontrolü: lokal boya, parça değişimi ve çizik/sürtme izlerini netleştir.",
            "Yetkili servis/servis kayıtlarını doğrula; periyodik bakımlar ve kampanya/geri çağırma var mı bak.",
            "Test sürüşünde rüzgâr sesi, trim sesleri ve fren performansını kontrol et.",
        ]
    elif (age is not None and age >= 10) or km >= 180_000:
        out += [
            "Motor: kompresyon/yağ kaçak kontrolü ve soğutma sistemi kontrolü yaptır.",
            "Şanzıman: soğuk/sıcak testte vuruntu/gecikme/titreme var mı kontrol et; bakım geçmişini sor.",
            "Alt takım: lift üstünde burçlar, amortisör, salıncak ve aks körüklerini kontrol ettir.",
        ]
    else:
        out += [
            "Rutin bakım kalemleri (yağ/filtreler, triger, fren) tarihlerini belgeyle doğrula.",
            "Lastik diş derinliği, balans ve fren performansını test sürüşünde değerlendir.",
        ]

    if fuel == "diesel":
        out.append("Dizel: DPF/EGR durumu ve enjektör/turbo kontrolü özellikle şehir içi kullanımda önemli.")
    elif fuel == "lpg":
        out.append("LPG: proje/ruhsat, montaj kalitesi, ayar ve subap durumu kontrol edilmeli.")
    elif fuel in ("hybrid","electric"):
        out.append("Hibrit/EV: batarya sağlığı ve servis altyapısı doğrulanmalı (hata kaydı taraması faydalı).")

    for item in base_list:
        if len(out) >= max_items:
            break
        if item not in out:
            out.append(item)
    return out[:max_items]


def build_premium_template(req: AnalyzeRequest, enriched: Dict[str, Any]) -> Dict[str, Any]:
    v = req.vehicle
    title = f"{v.year or ''} {v.make} {v.model}".strip() or "Premium Analiz"

    scores = compute_subscores(enriched, req)
    costs = enriched.get("costs", {}) or {}
    ins = enriched.get("insurance", {}) or {}
    taxes = enriched.get("taxes", {}) or {}
    fixed = enriched.get("fixed_costs", {}) or {}
    risk = enriched.get("risk", {}) or {}
    iq = enriched.get("info_quality", {}) or {}
    idx = (enriched.get("market", {}) or {}).get("indices", {}) or {}

    listed = costs.get("listed_price_try")
    maint_min, maint_max = int(costs.get("maintenance_yearly_try_min") or 0), int(costs.get("maintenance_yearly_try_max") or 0)
    fuel_min, fuel_max = int(costs.get("yearly_fuel_tr_min") or 0), int(costs.get("yearly_fuel_tr_max") or 0)
    fuel_mid = int(costs.get("yearly_fuel_tr_mid") or 0)

    mtv = (taxes.get("mtv") or {})
    insp = (fixed.get("inspection") or {})
    traffic = (ins.get("traffic") or {})
    kasko = (ins.get("kasko") or {})

    parts_av = int(idx.get("parts_availability_score_1_5", 3))
    parts_cost = int(idx.get("parts_cost_index_1_5", 3))
    service_net = int(idx.get("service_network_index_1_5", 3))
    resale_liq = int(idx.get("resale_liquidity_score_1_5", 3))

    checklist = build_buy_checklist(enriched, req, max_items=6)

    # Gerekçeler (skorların “dolu dolu” görünmesi için)
    reason_lines: List[str] = []
    age = risk.get("age")
    km = int(risk.get("mileage_km") or 0)
    segment_name = (enriched.get("segment", {}) or {}).get("name") or "-"
    fuel = str(req.vehicle.fuel or req.profile.fuel_preference or "").lower()
    usage = str((enriched.get("profile", {}) or {}).get("usage") or "mixed")
    yearly_km = int((enriched.get("profile", {}) or {}).get("yearly_km") or 15000)

    if age is not None:
        reason_lines.append(f"Yaş: **{age}** (model yılına göre hesaplandı).")
    if km:
        reason_lines.append(f"Km: **{_fmt_try(km)}**.")
    reason_lines.append(f"Segment: **{segment_name}**.")
    if fuel:
        reason_lines.append(f"Yakıt: **{fuel}**, kullanım: **{usage}**, yıllık km: **{_fmt_try(yearly_km)}**.")

    if iq.get("uncertainty"):
        reason_lines.append(f"Belirsizlik: **{iq.get('uncertainty')}** (eksikler: {', '.join(iq.get('missing_fields', [])[:6]) or 'yok'}).")

    # Son Özet (ekranın sonunda)
    final_summary: List[str] = []
    final_summary.append(f"Genel skor: **{_score10(scores['overall_100'])}/10** _(100’lük: {scores['overall_100']})_.")
    final_summary.append(f"Alt skorlar: mekanik **{_score10(scores['mechanical_100'])}/10**, elektronik **{_score10(scores['electronics_100'])}/10**, maliyet dengesi **{_score10(scores['economy_100'])}/10**, 2.el **{_score10(scores['resale_100'])}/10**, kişiye uygunluk **{_score10(scores['personal_fit_100'])}/10**.")
    if isinstance(listed, int) and listed > 0:
        final_summary.append(f"Yıllık toplam orta tahmin (bakım+yakıt+sigorta+vergi): ~**{_fmt_try(scores['annual_mid_try'])} TL/yıl** _(bandlar aşağıda)_.")

    # Markdown result
    lines: List[str] = []
    lines.append(f"## {title}")
    lines.append("")
    if (enriched.get("market", {}) or {}).get("profile_match") == "segment_estimate":
        lines.append("**Not:** Bu model için doğrudan profil verisi bulunamadı; aynı segment emsallerinden tahmini endeks/skor üretildi.")
        lines.append("")
    lines.append(f"**Bilgi seviyesi:** {iq.get('level','-')} (eksikler: {', '.join(iq.get('missing_fields', [])[:6]) or 'yok'})")
    lines.append("")

    lines.append("### 1) Oto Analiz Skoru")
    lines.append(f"**{_score10(scores['overall_100'])} / 10** _(100’lük: {scores['overall_100']})_")
    lines.append("")

    lines.append("### 2) Alt Skorlar")
    lines.append(f"- Mekanik: **{_score10(scores['mechanical_100'])} / 10**")
    lines.append(f"- Elektronik/Donanım: **{_score10(scores['electronics_100'])} / 10**")
    lines.append(f"- Maliyet Dengesi: **{_score10(scores['economy_100'])} / 10**")
    lines.append(f"- 2. El Likidite: **{_score10(scores['resale_100'])} / 10**")
    lines.append("")

    lines.append("### 3) Kişiye Uygunluk")
    lines.append(f"**{_score10(scores['personal_fit_100'])} / 10** _(100’lük: {scores['personal_fit_100']})_")
    lines.append(f"- Profil: **{_fmt_try(yearly_km)} km/yıl**, kullanım: **{usage}**, tercih: **{fuel or '-'}**")
    lines.append(f"- Segment: **{segment_name}**")
    if scores.get("personal_fit_reasons"):
        for r in scores["personal_fit_reasons"][:5]:
            lines.append(f"  - {r}")
    lines.append("")

    lines.append("### 4) Yıllık maliyet özeti (tahmini band)")
    lines.append(f"- Bakım: **{_fmt_try(maint_min)} – {_fmt_try(maint_max)} TL/yıl**")
    lines.append(f"  - Periyodik pay: ~**{_fmt_try(int(costs.get('maintenance_routine_yearly_est') or 0))} TL/yıl**")
    lines.append(f"  - Beklenmeyen gider payı: ~**{_fmt_try(int(costs.get('maintenance_risk_reserve_yearly_est') or 0))} TL/yıl**")
    lines.append(f"- Yakıt/enerji: **{_fmt_try(fuel_min)} – {_fmt_try(fuel_max)} TL/yıl** (orta: {_fmt_try(fuel_mid)} TL)")
    if mtv.get("ok"):
        lines.append(f"- MTV: **{_fmt_try(int(mtv.get('mtv_yearly_try_min') or 0))} – {_fmt_try(int(mtv.get('mtv_yearly_try_max') or 0))} TL/yıl**")
    if insp.get("ok"):
        lines.append(f"- Muayene yıllık ortalama: **~{_fmt_try(int(insp.get('annual_equivalent_try_2026_estimated') or 0))} TL/yıl**")
    if traffic.get("ok"):
        lines.append(f"- Trafik sigortası: **{_fmt_try(int(traffic.get('traffic_est_try_min') or 0))} – {_fmt_try(int(traffic.get('traffic_est_try_max') or 0))} TL/yıl**")
    if kasko.get("ok"):
        lines.append(f"- Kasko: **{_fmt_try(int(kasko.get('kasko_try_min') or 0))} – {_fmt_try(int(kasko.get('kasko_try_max') or 0))} TL/yıl**")
    lines.append(f"- Not: Yakıt türü ve kullanım şekli bazı kalemlerin takip sıklığını etkileyebilir.")
    lines.append("")

    lines.append("### 5) Parça, servis ve ikinci el endeksleri")
    lines.append(f"- Parça bulunabilirliği: **{parts_av}/5**")
    lines.append(f"- Parça maliyet endeksi: **{parts_cost}/5**")
    lines.append(f"- Servis ağı: **{service_net}/5**")
    lines.append(f"- 2. el likidite: **{resale_liq}/5**")
    lines.append("")

    lines.append("### 6) Kontrol listesi (kısa)")
    for c in checklist[:6]:
        lines.append(f"- {c}")
    lines.append("")

    lines.append("### 7) Notlar")
    for rr in reason_lines[:6]:
        lines.append(f"- {rr}")
    if risk.get("fuel_risk_comment"):
        lines.append(f"- Yakıt/kullanım notu: {risk.get('fuel_risk_comment')}")
    if risk.get("risk_notes"):
        for rn in (risk.get("risk_notes") or [])[:2]:
            lines.append(f"- {rn}")
    lines.append("")

    lines.append("### 8) Son Özet")
    for fs in final_summary[:4]:
        lines.append(f"- {fs}")
    lines.append("- Değerlendirmeyi netleştirmek için: **tramer**, **bakım/fatura geçmişi**, **OBD taraması** ve **test sürüşü** birlikte düşünülmelidir.")
    lines.append("")

    result_text = "\n".join(lines)

    out = {
        "scores": {
            "overall_100": scores["overall_100"],
            "mechanical_100": scores["mechanical_100"],
            "body_100": _clamp(scores["mechanical_100"] - 2, 0, 100),  # geriye uyumluluk için
            "economy_100": scores["economy_100"],
            "comfort_100": _clamp(70 + (4 if (enriched.get("segment", {}) or {}).get("code") in ("D_SEDAN","PREMIUM_D","E_SEGMENT") else 0), 0, 100),
            "family_use_100": _clamp(72 + (4 if (enriched.get("segment", {}) or {}).get("code") in ("C_SEDAN","C_SUV","D_SEDAN") else 0), 0, 100),
            "resale_100": scores["resale_100"],
            "electronics_100": scores["electronics_100"],
            "personal_fit_100": scores["personal_fit_100"],
        },
        "cost_estimates": {
            "yearly_total_mid_try": scores.get("annual_mid_try"),
            "yearly_maintenance_tr_min": maint_min,
            "yearly_maintenance_tr_max": maint_max,
            "yearly_fuel_tr_min": fuel_min,
            "yearly_fuel_tr_max": fuel_max,
            "notes": "Bandlar; segment + yaş/km + profil + (varsa) ilan fiyatı üzerinden tahmini üretilir.",
        },
        "preview": {
            "title": title,
            "price_tag": _price_tag(listed),
            "spoiler": "Skorlar + maliyet bandı + kişiye uygunluk + kısa kontrol listesi",
            "bullets": ["Genel & alt skorlar", "Kişiye uygunluk", "Yıllık maliyet bandı", "Kontrol listesi"],
        },
        "result": result_text,
    }
    return out


# =========================
# OPTIONAL LLM REWRITE (only wording)
# =========================
def call_llm_json(model_name: str, system_prompt: str, user_content: str) -> Optional[Dict[str, Any]]:
    if client is None:
        return None
    try:
        resp = client.chat.completions.create(
            model=model_name,
            response_format={"type": "json_object"},
            messages=[{"role":"system","content": system_prompt}, {"role":"user","content": user_content}],
        )
        content = resp.choices[0].message.content
        return json.loads(content) if isinstance(content, str) else content
    except Exception as e:
        print("LLM error:", e)
        return None

SYSTEM_PROMPT_PREMIUM_REWRITE = """
Sen Oto Analiz PREMIUM metinlerini daha doğal ve "kullanıcı diliyle" yazan bir asistansın.

Kurallar:
- Kesin yargı yok: 'alınır/alınmaz', 'sakın', 'tehlikeli' gibi ifadeler KULLANMA.
- Rakam UYDURMA veya değiştirmen yasak.
- Klişe/ansiklopedi gibi genel cümlelerden kaçın: her madde bu araç/profil verisine referans versin.
- Dil Türkçe.
- ÇIKTI SADECE JSON:

{
  "extra_summary_lines": ["...", "...", "..."]
}
""".strip()

def premium_analyze_impl(req: AnalyzeRequest) -> Dict[str, Any]:
    enriched = build_enriched_context(req)
    base = build_premium_template(req, enriched)

    # İstersen sadece en sona eklenecek 2-4 satırı LLM ile daha "insan" yazdırabiliriz.
    # Flutter kırmamak için result'a ek satır olarak ekliyoruz.
    rewrite_input = {
        "vehicle": req.vehicle.dict(),
        "profile": req.profile.dict(),
        "info_quality": enriched.get("info_quality"),
        "segment": enriched.get("segment"),
        "scores": base.get("scores"),
        "cost_estimates": base.get("cost_estimates"),
    }

    llm = call_llm_json(
        model_name=OPENAI_MODEL_PREMIUM,
        system_prompt=SYSTEM_PROMPT_PREMIUM_REWRITE,
        user_content=json.dumps(rewrite_input, ensure_ascii=False),
    )

    if isinstance(llm, dict) and isinstance(llm.get("extra_summary_lines"), list) and base.get("result"):
        extras = [str(x) for x in llm["extra_summary_lines"] if isinstance(x, str)]
        if extras:
            base["result"] += "\n\n**Ek Özet (kısa):**\n" + "\n".join([f"- {x}" for x in extras[:4]])

    return base


# =========================
# NORMAL / MANUAL / COMPARE / OTOBOT
# =========================
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
            "short_comment": "Sınırlı bilgiye göre genel bir değerlendirme üretildi.",
            "pros": ["Ekspertiz ve kayıtlar netleşince değerlendirme daha isabetli olur."],
            "cons": ["İlan detayları sınırlıysa skorlar daha geniş belirsizlikle oluşur."],
            "estimated_risk_level": "orta",
        },
        "preview": {"title": title, "price_tag": None, "spoiler": "Genel değerlendirme hazır.", "bullets": ["Tramer", "Bakım kayıtları", "Test sürüşü"]},
        "result": "Bu sonuç genel amaçlıdır. Tramer + bakım geçmişi + test sürüşü ile daha netleşir.",
    }

def call_llm_or_fallback(model_name: str, system_prompt: str, user_content: str, fallback_fn, req_obj):
    out = call_llm_json(model_name, system_prompt, user_content)
    if isinstance(out, dict):
        return out
    return fallback_fn(req_obj)


# =========================
# HEALTH
# =========================
@app.get("/")
async def root() -> Dict[str, Any]:
    return {"ok": True, "message": "Oto Analiz backend çalışıyor."}


# =========================
# ENDPOINTS
# =========================
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
        "hint": "Kesin yargı yok. Skor + kısa artı/eksi + belirsizlik notu üret."
    }
    return call_llm_or_fallback(OPENAI_MODEL_NORMAL, SYSTEM_PROMPT_NORMAL, json.dumps(user_content, ensure_ascii=False), fallback_normal, req)

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
        "hint": "Kesin yargı yok. Skor + bakım planı + kullanım profiline uygunluk notu üret."
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
    return {
        "better_overall": "left",
        "summary": "Karşılaştırma genel amaçlı üretildi; skorları netleştirmek için iki araçta da kayıt/ekspertiz gerekir.",
        "left_pros": ["Daha dengeli masraf/likidite profili olabilir."],
        "left_cons": ["Detaylar netleşmeden skorlar geniş banttır."],
        "right_pros": ["Doğru geçmişle mantıklı alternatif olabilir."],
        "right_cons": ["Masraf dağılımı daha değişken hissedilebilir."],
        "use_cases": {
            "family_use": "Aile kullanımında alan/konfor ve masraf dengesi belirleyicidir.",
            "long_distance": "Uzun yolda bakım geçmişi ve lastik/fren durumu kritiktir.",
            "city_use": "Şehir içinde tüketim ve şanzıman karakteri belirleyicidir."
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
        "answer": "Bütçe, yıllık km ve kullanım tipine göre segment seçmek en mantıklısıdır. Kayıt/ekspertiz ile netleşir.",
        "suggested_segments": ["C-sedan", "C-SUV", "B-Hatch"],
        "example_models": ["Toyota Corolla", "Renault Megane", "Honda Civic", "Hyundai Tucson"]
    }
