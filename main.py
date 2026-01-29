import os
import json
import re
import math
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple
import base64
import uuid
from io import BytesIO
from urllib.parse import quote as urlquote
import requests


from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
# The `dotenv` package is optional. When it is not installed in the
# environment, importing it will raise a `ModuleNotFoundError`. To
# avoid crashing at import time (which would lead to a 500 error when
# the FastAPI application starts), we try to import `load_dotenv` and
# fall back to a no-op function if it cannot be imported.
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    # Define a dummy `load_dotenv` so calls to it are harmless when
    # `python-dotenv` is not available. It accepts arbitrary arguments
    # but performs no action.
    def load_dotenv(*args: Any, **kwargs: Any) -> None:  # type: ignore
        return None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


# =========================================================
# OPTIONAL DEPS (PIL + Firebase Admin)
# =========================================================
try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = None  # type: ignore
    ImageDraw = None  # type: ignore
    ImageFont = None  # type: ignore

try:
    import firebase_admin
    from firebase_admin import credentials as fb_credentials
    from firebase_admin import firestore as fb_firestore
    from firebase_admin import storage as fb_storage
except Exception:
    firebase_admin = None  # type: ignore
    fb_credentials = None  # type: ignore
    fb_firestore = None  # type: ignore
    fb_storage = None  # type: ignore


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

# =========================================================
# SS OCR (AI Vision) – screenshot -> structured fields
# =========================================================
SS_OCR_MODEL = os.getenv("SS_OCR_MODEL", "gpt-5-mini")
try:
    SS_OCR_MAX_IMAGE_MB = float(os.getenv("SS_OCR_MAX_IMAGE_MB", "8"))
except Exception:
    SS_OCR_MAX_IMAGE_MB = 8.0
try:
    SS_OCR_MAX_SIDE = int(os.getenv("SS_OCR_MAX_SIDE", "1600"))
except Exception:
    SS_OCR_MAX_SIDE = 1600

_BODY_TYPES = [
    "Hatchback",
    "Sedan",
    "Station Wagon",
    "SUV",
    "Coupe",
    "Cabrio",
    "MPV",
    "Pick-up",
    "Van",
    "Diğer",
]

DATA_DIR = os.getenv("DATA_DIR", "data")  # bundle içinde ./data


# =========================================================
# APP
# =========================================================
app = FastAPI(title="Oto Analiz Backend")


# =========================================================
# SS OCR endpoint (Full AI) – kullanıcı SS seçer, alanlar otomatik dolar
# =========================================================
def _to_data_url(image_bytes: bytes, mime: str) -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def _maybe_resize_image(image_bytes: bytes, mime: str) -> Tuple[bytes, str]:
    """Resize the image to reduce cost/latency. If PIL is unavailable, return as-is."""
    if Image is None:
        return image_bytes, mime
    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        w, h = img.size
        max_side = max(w, h)
        if SS_OCR_MAX_SIDE and max_side > SS_OCR_MAX_SIDE:
            scale = SS_OCR_MAX_SIDE / float(max_side)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            img = img.resize((new_w, new_h))
        out = BytesIO()
        # Always encode to JPEG for smaller payload (vision still works well)
        img.save(out, format="JPEG", quality=85, optimize=True)
        return out.getvalue(), "image/jpeg"
    except Exception:
        return image_bytes, mime

def _call_vision_json(model_name: str, system_prompt: str, user_text: str, image_data_url: str) -> Dict[str, Any]:
    if client is None:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY missing or OpenAI SDK not available on server.")
    try:
        resp = client.chat.completions.create(
            model=model_name,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                },
            ],
        )
        txt = resp.choices[0].message.content
        if not isinstance(txt, str):
            raise ValueError("Model JSON output is not a string")
        return json.loads(txt)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SS OCR failed: {e}")

def _normalize_body_type(x: Optional[str]) -> Optional[str]:
    if not x:
        return None
    s = str(x).strip().lower()
    # quick Turkish/eng synonyms
    if "hatch" in s or s in {"hb", "h/b", "hatcback"}:
        return "Hatchback"
    if "sedan" in s:
        return "Sedan"
    if "station" in s or "wagon" in s or "variant" in s or "sw" in s:
        return "Station Wagon"
    if "suv" in s or "crossover" in s or "cros" in s:
        return "SUV"
    if "coupe" in s or "kup" in s:
        return "Coupe"
    if "cabrio" in s or "convert" in s:
        return "Cabrio"
    if "mpv" in s or "minivan" in s:
        return "MPV"
    if "pick" in s or "pickup" in s or "kamyonet" in s:
        return "Pick-up"
    if "van" in s or "panelvan" in s or "minib" in s:
        return "Van"
    if "diğer" in s or "diger" in s or "other" in s:
        return "Diğer"
    # already exact?
    for bt in _BODY_TYPES:
        if s == bt.lower():
            return bt
    return None

@app.post("/ss_ocr")
async def ss_ocr(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Screenshot OCR via full AI (vision). Returns structured fields for auto-fill."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Lütfen bir görsel (image/*) dosyası yükleyin.")
    raw = await file.read()
    max_bytes = int(SS_OCR_MAX_IMAGE_MB * 1024 * 1024)
    if max_bytes > 0 and len(raw) > max_bytes:
        raise HTTPException(status_code=400, detail=f"Görsel çok büyük. Max {SS_OCR_MAX_IMAGE_MB}MB.")

    resized_bytes, mime = _maybe_resize_image(raw, file.content_type)
    data_url = _to_data_url(resized_bytes, mime)

    system_prompt = (
        "Sen Türkiye ikinci el araç ilanı ekran görüntülerinden alan çıkaran bir asistansın. "
        "SADECE geçerli JSON döndür. Ekstra metin, açıklama, markdown YOK. "
        "Emin olmadığın alanları null bırak. Uydurma yapma. "
        "Fiyat TRY integer (sadece rakam) olmalı. Km integer (sadece rakam) olmalı. Yıl integer olmalı."
    )

    user_text = (
        "Bu bir araç ilanı ekran görüntüsü. Aşağıdaki alanları çıkar ve JSON olarak döndür:\n"
        "{\n"
        "  \"brand\": string|null,\n"
        "  \"model\": string|null,\n"
        "  \"series\": string|null,\n"
        "  \"year\": integer|null,\n"
        "  \"km\": integer|null,\n"
        "  \"price_try\": integer|null,\n"
        "  \"power\": string|null,\n"
        "  \"fuel\": string|null,\n"
        "  \"transmission\": string|null,\n"
        "  \"body_type\": string|null,\n"
        "  \"location\": string|null,\n"
        "  \"title\": string|null\n"
        "}\n"
        "body_type sadece şu listeden biri olsun: "
        + ", ".join(_BODY_TYPES)
        + ". Eşleşmiyorsa null bırak."
    )

    data = _call_vision_json(SS_OCR_MODEL, system_prompt, user_text, data_url)

    # Normalize and sanitize
    out: Dict[str, Any] = {
        "brand": data.get("brand"),
        "model": data.get("model"),
        "series": data.get("series"),
        "year": data.get("year"),
        "km": data.get("km"),
        "price_try": data.get("price_try"),
        "power": data.get("power"),
        "fuel": data.get("fuel"),
        "transmission": data.get("transmission"),
        "body_type": _normalize_body_type(data.get("body_type")),
        "location": data.get("location"),
        "title": data.get("title"),
    }
    return {"ok": True, "data": out, "model": SS_OCR_MODEL}


# =========================================================
# OTOBOT (Mini V1) – external module (otobot_min.py)
# =========================================================
# Not: Bu import hata verirse backend tamamen düşmesin diye try/except.
# otobot_min.py dosyası main.py ile aynı klasörde olmalı.
try:
    from otobot_min import router as otobot_router  # type: ignore
    app.include_router(otobot_router)
except Exception as _e:
    print("OtoBot router yüklenemedi (otobot_min.py eksik olabilir):", _e)

    from pydantic import BaseModel, Field

    class _OtoBotFallbackReq(BaseModel):
        mode: str = Field("recommend", description="recommend | guide")
        message: str = Field("", description="Kullanıcı mesajı / ihtiyacı")

    @app.post("/otobot")
    async def otobot_fallback(req: _OtoBotFallbackReq) -> Dict[str, Any]:
        # Sunucu çökmesin; otobot_min.py gelene kadar açıklayıcı cevap dön.
        return {
            "ok": False,
            "error": "OtoBot modülü yüklenemedi. Backend klasörüne otobot_min.py dosyasını ekleyip tekrar deploy et.",
            "hint": "otobot_min.py main.py ile aynı klasörde olmalı. Repo'ya commit edip Render'a pushla.",
            "received": {"mode": req.mode, "message_preview": (req.message or '')[:120]},
        }


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

    # ✅ Şehir bilgisi (isteğe bağlı): İstanbul seçtiyse bunu yazar
    city: Optional[str] = None        # "İstanbul" / "34" vb.
    city_code: Optional[str] = None   # "34" vb. (plaka kodu)

    # ✅ Vites tercihi (opsiyonel)
    transmission_preference: Optional[str] = None  # auto / manual / any

    class Config:
        extra = "allow"


class Vehicle(BaseModel):
    make: str = ""
    model: str = ""
    year: Optional[int] = Field(None, ge=1980, le=2035)
    mileage_km: Optional[int] = Field(None, ge=0)
    fuel: Optional[str] = None        # gasoline / diesel / lpg / hybrid / electric

    # ✅ İlan/SS’den gelebilir
    transmission: Optional[str] = None  # automatic / manual

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
    analysis_mode: Optional[str] = Field(None, description="quick|premium")

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


def _fmt_int(n: Optional[int]) -> str:
    """
    Format a numeric value with thousands separators for readability.

    This helper mirrors `_fmt_try` but does not treat `None` specially. It
    expects a value that can be cast to an integer; if formatting fails,
    it returns a dash. Example: `_fmt_int(1234567)` -> "1.234.567".
    """
    try:
        return f"{int(n):,}".replace(",", ".")
    except Exception:
        return "-"


def _clamp(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, n))


def _usage_tr(u: str) -> str:
    u = (u or "").strip().lower()
    if u == "city":
        return "şehir içi"
    if u == "highway":
        return "uzun yol"
    return "karma kullanım"  # mixed default


def _fuel_tr(f: str) -> str:
    f = (f or "").strip().lower()
    mp = {
        "gasoline": "benzin",
        "diesel": "dizel",
        "lpg": "lpg",
        "hybrid": "hibrit",
        "electric": "elektrik",
    }
    return mp.get(f, f or "-")


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
    m = re.search(r"(?<!\d)(0\.\d|1\.\d|2\.\d|3\.\d|4\.\d|5\.\d|6\.\d)(?!\d)", t)
    if not m:
        return None
    try:
        liters = float(m.group(1))
        cc = int(round(liters * 1000))
        if 600 <= cc <= 6500:
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
    if "4x4" in t or "awd" in t or "quattro" in t or "xdrive" in t or "4matic" in t:
        tags.append("awd_optional")
    if "turbo" in t or "tce" in t or "tsi" in t or "ecoboost" in t:
        tags.append("turbo_small")

    # Performance cues
    if re.search(r"\brs\d\b", t) or "rs " in t or "amg" in t or re.search(r"\bc ?63\b", t) or re.search(r"\bm\d\b", t):
        tags.append("performance")

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

# ✅ Yeni: Şehir trafik yoğunluğu (opsiyonel veri pack)
CITY_CONGESTION: Dict[str, Any] = _load_json(_dp("city_congestion_tr_tomtom_2024_citycenter.json"), {})


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
# CITY + TRANSMISSION HELPERS
# =========================================================
def _zfill_plate(code: Any) -> Optional[str]:
    if code is None:
        return None
    try:
        c = str(int(str(code).strip()))
        if 1 <= int(c) <= 81:
            return c.zfill(2)
    except:
        pass
    s = str(code).strip()
    if s.isdigit():
        try:
            n = int(s)
            if 1 <= n <= 81:
                return str(n).zfill(2)
        except:
            pass
    return None


def _resolve_plate_city_code(req: AnalyzeRequest) -> str:
    # 1) context üst öncelik
    ctx = req.context or {}
    code = _zfill_plate(ctx.get("plate_city_code") or ctx.get("city_code") or ctx.get("plate"))
    if code:
        return code

    # 2) profile.city_code
    p = req.profile or Profile()
    code = _zfill_plate(getattr(p, "city_code", None))
    if code:
        return code

    # 3) profile.city sayıysa
    code = _zfill_plate(getattr(p, "city", None))
    if code:
        return code

    # 4) profile.city isimse TRAFFIC_CAPS içinden eşleştir
    city_name = getattr(p, "city", None)
    if isinstance(city_name, str) and city_name.strip():
        cities = (TRAFFIC_CAPS or {}).get("cities") or {}
        target = _norm(city_name)
        for k, v in cities.items():
            nm = v.get("name")
            if nm and _norm(nm) == target:
                kk = _zfill_plate(k)
                if kk:
                    return kk

    # default İstanbul
    return "34"


def _resolve_city_name_by_code(code: str) -> Optional[str]:
    cities = (TRAFFIC_CAPS or {}).get("cities") or {}
    c = _zfill_plate(code) or code
    info = cities.get(c)
    if isinstance(info, dict) and info.get("name"):
        return info["name"]
    return None


def _infer_transmission(req: AnalyzeRequest) -> Optional[str]:
    # 1) Vehicle alanı
    if getattr(req.vehicle, "transmission", None):
        t = _norm(str(req.vehicle.transmission))
        if "auto" in t or "otom" in t:
            return "automatic"
        if "man" in t:
            return "manual"

    # 2) context
    ctx = req.context or {}
    for k in ("transmission", "vites", "gearbox", "sanziman"):
        if ctx.get(k):
            t = _norm(str(ctx.get(k)))
            if "otom" in t or "auto" in t or "dsg" in t or "cvt" in t or "dct" in t or "ptronic" in t or "tiptronic" in t:
                return "automatic"
            if "man" in t:
                return "manual"

    # 3) metin tarama
    blob = f"{req.ad_description or ''} {json.dumps(ctx, ensure_ascii=False)} {req.vehicle.make} {req.vehicle.model}"
    t = _norm(blob)
    if any(x in t for x in ["otomatik", "automatic", "dsg", "s tronic", "s-tronic", "cvt", "dct", "edc", "powershift", "tiptronic", "ptronic", "zf"]):
        return "automatic"
    if any(x in t for x in ["manuel", "manual"]):
        return "manual"

    return None


def get_city_congestion_context(req: AnalyzeRequest) -> Dict[str, Any]:
    code = _resolve_plate_city_code(req)
    name = _resolve_city_name_by_code(code) or (getattr(req.profile, "city", None) if req.profile else None)

    pack_cities = (CITY_CONGESTION or {}).get("cities") or {}
    row = pack_cities.get(code)

    if not isinstance(row, dict):
        return {
            "ok": False,
            "city_code": code,
            "city_name": name,
            "source": (CITY_CONGESTION or {}).get("source"),
            "note": "Bu şehir için trafik yoğunluğu datası yok; genel yorum üretilecek."
        }

    avg = row.get("avg_congestion_level_pct")
    morning = row.get("morning_peak_pct")
    evening = row.get("evening_peak_pct")

    def _label(x: Optional[int]) -> str:
        if not isinstance(x, int):
            return "bilinmiyor"
        if x >= 45:
            return "çok yoğun"
        if x >= 35:
            return "yoğun"
        if x >= 25:
            return "orta"
        return "rahat"

    return {
        "ok": True,
        "city_code": code,
        "city_name": row.get("name") or name,
        "avg_pct": int(avg) if isinstance(avg, int) else None,
        "morning_peak_pct": int(morning) if isinstance(morning, int) else None,
        "evening_peak_pct": int(evening) if isinstance(evening, int) else None,
        "label": _label(int(avg) if isinstance(avg, int) else None),
        "source": (CITY_CONGESTION or {}).get("source"),
        "note": None,
    }


def build_city_fit_lines(req: AnalyzeRequest, enriched: Dict[str, Any]) -> List[str]:
    ctx = get_city_congestion_context(req)
    prof = enriched.get("profile", {}) or {}
    usage = prof.get("usage", "mixed")
    yearly_km = int(prof.get("yearly_km", 15000) or 15000)

    fuel = (req.vehicle.fuel or req.profile.fuel_preference or "").lower().strip()
    trn = _infer_transmission(req)

    out: List[str] = []

    # şehir satırı
    if ctx.get("city_name"):
        if ctx.get("ok"):
            out.append(
                f"- Şehir: **{ctx['city_name']}** → trafik yoğunluğu **{ctx.get('label','-')}** "
                f"(ortalama ~%{ctx.get('avg_pct','-')}, akşam pik ~%{ctx.get('evening_peak_pct','-')})."
            )
        else:
            out.append(f"- Şehir: **{ctx['city_name']}** → trafik yoğunluğu verisi yok; genel şehir içi mantığıyla yorumlandı.")

    # ağır trafik var mı?
    heavy_city = False
    if ctx.get("ok") and isinstance(ctx.get("evening_peak_pct"), int) and ctx["evening_peak_pct"] >= 70:
        heavy_city = True
    if usage == "city":
        heavy_city = True

    # vites yorumu
    if heavy_city:
        if trn == "manual":
            out.append("- Trafik/stop-go senaryoda **manuel vites yorucu** olabilir; otomatik/yarı otomatik konforu ciddi artırır.")
        elif trn == "automatic":
            out.append("- Trafik yoğun kullanımda **otomatik/yarı otomatik** tercih, konfor ve günlük kullanım kolaylığı sağlar.")
        else:
            out.append("- Vites tipi net değil; yoğun trafikte otomatik/yarı otomatik genelde daha rahat bir tercih olur.")

    # yakıt yorumu
    if heavy_city:
        if fuel == "diesel" and yearly_km < 15000:
            out.append("- Yoğun şehir içi + düşük/orta km’de **dizelde DPF/EGR** tarafı daha hassas olabilir (kısa mesafe arttıkça risk büyür).")
        if fuel in ("hybrid", "electric"):
            out.append("- **Hibrit/EV**, stop-go trafikte tüketim avantajını ve rejenerasyon faydasını daha belirgin hissettirir.")
        if fuel == "lpg":
            out.append("- **LPG** şehir içinde ekonomik olabilir; ancak montaj/ayar kalitesi ve periyodik kontrol daha kritik hale gelir.")

    # kullanım tipine bağla
    if usage == "highway":
        out.append("- Uzun yol ağırlıklı kullanımda tüketim daha stabil olur; bakım disiplini ve lastik/fren durumu öne çıkar.")
    elif usage == "city":
        out.append("- Şehir içi ağırlıkta fren/lastik/alt takım yıpranması daha hızlı olabilir; buna göre bütçe payı bırakmak mantıklı.")

    return out[:6]


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
    # ✅ Yeni: RS / AMG / M / S / Turbo performans sınıfı
    "LUX_PERFORMANCE": {
        "name": "Lüks performans (RS / AMG / M / S / V)",
        "maintenance_yearly_range": (65000, 190000),
        "insurance_level": "çok yüksek",
        "notes": [
            "Bu sınıfta masrafın ana farkı: yüksek performanslı fren/lastik, karmaşık elektronik, pahalı işçilik ve yüksek kasko primi.",
            "Yaş/km arttıkça turbo/soğutma/şanzıman ve diferansiyel gibi kalemlerin maliyet etkisi büyür.",
        ],
    },
    # ✅ Yeni: Ferrari / Lamborghini / McLaren gibi supersport
    "SUPER_SPORT": {
        "name": "Supercar / supersport",
        "maintenance_yearly_range": (120000, 450000),
        "insurance_level": "çok yüksek (özel)",
        "notes": [
            "Parça temini, işçilik ve sarf kalemleri çok pahalı olabilir; servis/uzman erişimi sınırlı kalabilir.",
            "Bu sınıfta belirsizlik doğal olarak daha yüksek; geçmiş/servis kaydı ve kapsamlı kontrol kritik.",
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
# SEGMENT DETECTION (✅ güçlendirildi)
# =========================================================
def _is_supersport(s: str) -> bool:
    # Markalar + açık supersport ipuçları
    if any(k in s for k in ["ferrari", "lamborghini", "mclaren", "koenigsegg", "bugatti", "pagani", "aston martin valkyrie"]):
        return True
    if any(k in s for k in ["huracan", "aventador", "sf90", "488", "812", "f8", "roma", "portofino", "720s", "765lt", "p1", "senna", "chiron", "veyron"]):
        return True
    return False


def _is_lux_performance(s: str) -> bool:
    # Audi RS, BMW M, Mercedes AMG (özellikle 63), Porsche (911/GT), vb.
    if re.search(r"\brs\s?\d\b", s) or re.search(r"\brs\d\b", s):
        return True
    if re.search(r"\bm\s?\d\b", s) or re.search(r"\bm\d\b", s):  # m5, m3, m2...
        return True
    if "amg" in s:
        return True
    if re.search(r"\bc\s?63\b", s) or re.search(r"\be\s?63\b", s) or re.search(r"\bs\s?63\b", s):
        return True
    if any(k in s for k in ["c63", "e63", "s63", "cla45", "a45", "g63"]):
        return True
    if any(k in s for k in ["rs7", "rs6", "rs5", "rs3", "rsq8", "rsq3", "rs4"]):
        return True
    if any(k in s for k in ["m5", "m3", "m4", "m2", "x5m", "x6m", "m8"]):
        return True
    if any(k in s for k in ["porsche 911", "911", "gt3", "gt2", "turbo s", "taycan turbo", "panamera turbo"]):
        return True
    if any(k in s for k in ["bmw alpina", "alpina", "audi s8", "audi s6", "audi s7", "audi s5", "mercedes s500", "s580", "e53", "c43"]):
        return True
    return False


def detect_segment(make: str, model: str) -> str:
    s = _norm(f"{make} {model}")

    # 1) supersport
    if _is_supersport(s):
        return "SUPER_SPORT"

    # 2) lüks performans
    if _is_lux_performance(s):
        return "LUX_PERFORMANCE"

    # 3) premium markalar (normal premium)
    if any(k in s for k in ["bmw", "mercedes", "audi", "volvo", "lexus", "range rover", "land rover", "porsche"]):
        return "PREMIUM_D"

    # 4) yaygın segment heuristics
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

        if any(m in target for m in ["bmw", "mercedes", "audi", "volvo", "porsche"]) and \
           any(m in key for m in ["bmw", "mercedes", "audi", "volvo", "porsche"]):
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
            "LUX_PERFORMANCE": (2501, 5000),
            "SUPER_SPORT": (3501, 6500),
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

    city_code = _resolve_plate_city_code(req)
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
        "LUX_PERFORMANCE": (0.040, 0.110),
        "SUPER_SPORT": (0.060, 0.160),
        "E_SEGMENT": (0.035, 0.095),
    }.get(segment_code, (0.022, 0.055))

    age_mult = 1.0
    if age is not None:
        if age >= 15:
            age_mult = 1.45
        elif age >= 10:
            age_mult = 1.28
        elif age >= 6:
            age_mult = 1.12

    km_mult = 1.0
    if mileage_km >= 250_000:
        km_mult = 1.30
    elif mileage_km >= 180_000:
        km_mult = 1.18
    elif mileage_km >= 120_000:
        km_mult = 1.10

    rmin = base[0] * age_mult * km_mult
    rmax = base[1] * age_mult * km_mult

    kmin = int(listed_price * rmin)
    kmax = int(listed_price * rmax)

    # caps by segment
    max_cap_ratio = 0.12
    if segment_code == "LUX_PERFORMANCE":
        max_cap_ratio = 0.16
    if segment_code == "SUPER_SPORT":
        max_cap_ratio = 0.22

    kmax = min(kmax, int(listed_price * max_cap_ratio))
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
            age_mult = 2.1 if segment_code in ("LUX_PERFORMANCE", "SUPER_SPORT") else 1.8
        elif age >= 10:
            age_mult = 1.55 if segment_code in ("LUX_PERFORMANCE", "SUPER_SPORT") else 1.4
        elif age >= 6:
            age_mult = 1.20 if segment_code in ("LUX_PERFORMANCE", "SUPER_SPORT") else 1.15

    km_mult = 1.0
    if mileage >= 250_000:
        km_mult = 2.0 if segment_code in ("LUX_PERFORMANCE", "SUPER_SPORT") else 1.8
    elif mileage >= 180_000:
        km_mult = 1.6 if segment_code in ("LUX_PERFORMANCE", "SUPER_SPORT") else 1.4
    elif mileage >= 120_000:
        km_mult = 1.25 if segment_code in ("LUX_PERFORMANCE", "SUPER_SPORT") else 1.2

    usage_mult = 1.0
    if p.usage == "city":
        usage_mult = 1.18 if segment_code in ("LUX_PERFORMANCE", "SUPER_SPORT") else 1.15
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
            "LUX_PERFORMANCE": (0.030, 0.14),
            "SUPER_SPORT": (0.040, 0.20),
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
            "LUX_PERFORMANCE": 240_000,
            "SUPER_SPORT": 520_000,
            "E_SEGMENT": 160_000,
        }
        cap = caps.get(segment_code, 60_000)
        maint_max = min(maint_max, cap)

        if maint_min > maint_max:
            maint_min = int(maint_max * 0.7)

    mid_maint = int((maint_min + maint_max) / 2) if maint_max else maint_min
    routine_est = int(mid_maint * 0.62) if segment_code in ("LUX_PERFORMANCE", "SUPER_SPORT") else int(mid_maint * 0.65)
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
        "LUX_PERFORMANCE": (42_000, 105_000),
        "SUPER_SPORT": (65_000, 180_000),
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

    if segment_code in ("PREMIUM_D", "E_SEGMENT", "LUX_PERFORMANCE", "SUPER_SPORT") and ((age and age > 10) or mileage > 180_000):
        risk_notes.append("Premium/performance sınıfta yaşlı/yüksek km araçların büyük masraf kalemleri pahalı olabilir.")

    if segment_code in ("LUX_PERFORMANCE", "SUPER_SPORT"):
        risk_notes.append("Performans sınıfta fren/lastik/soğutma/şanzıman gibi kalemler daha pahalı ve daha kritik olabilir.")

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
            "LUX_PERFORMANCE": (10.5, 18.0),
            "SUPER_SPORT": (12.0, 22.0),
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
            "city": getattr(p, "city", None),
            "city_code": getattr(p, "city_code", None),
            "transmission_preference": getattr(p, "transmission_preference", None),
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
# PREMIUM: EXTRA "DOLU DOLU" HELPERS
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


def infer_target_user_line(segment_label: str, make: str = "", model: str = "") -> str:
    """Short 'who is this car for?' line used in both quick and premium reports."""
    sl = (segment_label or "").lower()
    mk = (make or "").lower()

    premium_brands = {
        "audi","bmw","mercedes","porsche","lexus","jaguar","land rover",
        "maserati","bentley","lamborghini","ferrari","rolls-royce","cadillac"
    }

    if any(k in sl for k in ("premium", "perform", "sport", "super", "lux")) or mk in premium_brands:
        return ("Bu tip araçlar daha çok performans/konforu bir arada isteyen ve "
                "masraf bütçesi yüksek kullanıcıya uygundur.")
    if any(k in sl for k in ("economy", "ekonomi", "compact", "şehir", "b-segment", "b segment", "c-segment", "c segment")):
        return ("Bu tip araçlar daha çok ekonomik/sorunsuzluk odaklı, şehir içi kullanımı yoğun "
                "ve bütçesi kontrollü kullanıcıya uygundur.")
    if any(k in sl for k in ("family", "aile", "suv", "crossover", "mpv", "station", "wagon")):
        return ("Bu tip araçlar daha çok aile kullanımı, geniş iç hacim ve konfor isteyen kullanıcıya uygundur; "
                "düzenli bakım geçmişi kritik olur.")

    return ("Bu araç; dengeli kullanım isteyen, masraf toleransı orta seviyede olan kullanıcıya daha uygundur.")


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

    if seg in ("PREMIUM_D", "E_SEGMENT", "LUX_PERFORMANCE", "SUPER_SPORT") and prof.get("yearly_km_band") == "düşük":
        score -= 8 if seg in ("LUX_PERFORMANCE", "SUPER_SPORT") else 6

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
    if seg in ("LUX_PERFORMANCE", "SUPER_SPORT"):
        score -= 9
    if age >= 10:
        score -= 6
    if age >= 15:
        score -= 6
    return _clamp(int(score), 0, 100)


def build_score_because(req: AnalyzeRequest, enriched: Dict[str, Any], overall: int, uncertainty: Dict[str, Any]) -> Dict[str, Any]:
    """
    Skor için kısa gerekçe (bayraksız, premium ton).
    Döner: {"because":"...", "critical":["...","..."]}
    """
    prof = enriched.get("profile", {}) or {}
    market = enriched.get("market", {}) or {}
    seg_code = (enriched.get("segment", {}) or {}).get("code", "C_SEDAN")

    fuel = (req.vehicle.fuel or req.profile.fuel_preference or "").lower().strip()
    usage = (prof.get("usage") or "mixed").lower().strip()
    match = (market.get("profile_match") or "")

    risk = enriched.get("risk", {}) or {}
    age = risk.get("age")
    km = int(risk.get("mileage_km") or 0)

    idx = (market.get("indices") or {})
    parts_av = int(idx.get("parts_availability_score_1_5", 3))
    resale_liq = int(idx.get("resale_liquidity_score_1_5", 3))

    plus: List[str] = []
    critical: List[str] = []

    # Belirsizlik
    if (uncertainty or {}).get("level") == "düşük":
        plus.append("bilgi seviyesi yüksek olduğu için belirsizlik düşük")
    elif (uncertainty or {}).get("level") == "orta":
        plus.append("bilgi seviyesi orta; bazı kalemler netleşirse skor daha güvenilir olur")
    else:
        critical.append("belirsizlik yüksek; tramer + servis kaydı + OBD ile netleştirmek önemli")

    # Segment yaygınlık / parça
    if parts_av >= 4 and seg_code in ("B_HATCH", "C_SEDAN", "C_SUV", "D_SEDAN"):
        plus.append("segment/parça-usta erişimi görece rahat")
    elif seg_code in ("PREMIUM_D", "E_SEGMENT", "LUX_PERFORMANCE", "SUPER_SPORT"):
        critical.append("premium/performance sınıfta parça-işçilik maliyeti daha yüksek eğilimli")

    # 2.el
    if resale_liq >= 4:
        plus.append("2. el likidite tarafı güçlü eğilimli")

    # Yakıt türü
    if fuel in ("hybrid", "electric"):
        plus.append("şehir içi stop-go senaryoda hibrit/EV tüketim avantajı görülebilir")
        critical.append("batarya sağlığını rapor/servis kaydıyla doğrulamak kritik")
    if fuel == "diesel" and usage == "city" and int(prof.get("yearly_km", 15000) or 15000) < 15000:
        critical.append("şehir içi + düşük/orta km’de dizelde DPF/EGR riski artabilir")

    # Yaş/km
    if (age is not None and age >= 10) or km >= 180_000:
        critical.append("yaş/km yükseldikçe büyük bakım riski artabilir")

    # Profil match
    if match == "segment_estimate":
        critical.append("bu model için direkt profil yok; emsal segmentten tahmin kullanıldı")

    # Performance özel kritik
    if seg_code in ("LUX_PERFORMANCE", "SUPER_SPORT"):
        critical.append("performans sınıfta fren/lastik/soğutma/şanzıman gibi kalemler pahalı; ekspertiz daha kritik")

    plus = [p for p in plus if p]
    critical = [c for c in critical if c]

    because_parts = []
    if plus:
        because_parts.append(", ".join(plus[:3]))
    because = "Skor; " + ("; ".join(because_parts) if because_parts else "mevcut verilere göre dengeli bir tabloya işaret ediyor") + "."

    return {"because": because, "critical": critical[:3]}


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
        if seg in ("PREMIUM_D", "E_SEGMENT", "LUX_PERFORMANCE", "SUPER_SPORT"):
            reasons.append("Premium/performance sınıfta OEM parça ve işçilik maliyeti genelde daha yüksek olur; stok/termin etkilenebilir.")
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

    return {
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

    # performans sınıfta etiket daha temkinli
    if segment_code in ("LUX_PERFORMANCE", "SUPER_SPORT"):
        if listed < 1_500_000:
            label = "Normal"
        elif listed < 3_000_000:
            label = "Yüksek"
        else:
            label = "Yüksek"
    else:
        if listed < 500_000:
            label = "Uygun"
        elif listed < 1_200_000:
            label = "Normal"
        else:
            label = "Yüksek"

    comment = "Değer yorumu; segment, yaş/km ve ilan detay düzeyine göre *yaklaşık* konumlandırma sağlar. Kesin piyasa fiyatı değildir."
    if segment_code in ("PREMIUM_D", "E_SEGMENT", "LUX_PERFORMANCE", "SUPER_SPORT"):
        comment += " Premium/performance sınıfta aynı fiyat bandında bile masraf/riske bağlı değer algısı çok değişebilir."
    if "ilan_aciklamasi_kisa" in missing:
        comment += " İlan açıklaması kısa olduğu için pazarlık argümanları daha çok ekspertiz sonucuna dayanmalı."

    args = [
        "Ekspertizde çıkan masrafları (lastik, fren, alt takım, bakım) tek tek fiyatlandırıp pazarlık maddesi yap.",
        "Bakım kayıtları ve faturalar yoksa: yakın vadeli bakım bütçesini (yağ/filtre/sıvılar) argümanlaştır.",
        "Tramer toplamı + parça değişimi bilgilerini (şasi/podye kontrolüyle birlikte) teyit edip belirsizliği pazarlıkta kullan.",
    ]
    if segment_code in ("LUX_PERFORMANCE", "SUPER_SPORT"):
        args.insert(0, "Performans sınıfta fren/lastik seti, disk-balata ve soğutma kalemlerinin fiyatını netleştirip pazarlığa çevir.")

    return {
        "ok": True,
        "label": label,
        "comment": comment,
        "negotiation_args": args[:3],
        "disclaimer": "Bu bölüm kesin hüküm içermez; sadece pazarlık yaklaşımı ve belirsizlik yönetimi içindir."
    }


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

    seg_code = (enriched.get("segment", {}) or {}).get("code")
    if seg_code in ("LUX_PERFORMANCE", "SUPER_SPORT"):
        warnings.append("Performans sınıfta fren/lastik ve soğutma sistemi kondisyonu (kaçak, radyatör, pompa, termostat) özellikle kritik.")
        warnings.append("Yük altında test sürüşü ve OBD/log kontrolü (misfire, knock, boost, sıcaklık) raporu çok değerli olur.")

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

    seg_code = (enriched.get("segment", {}) or {}).get("code")
    if seg_code in ("LUX_PERFORMANCE", "SUPER_SPORT"):
        checklist.insert(0, "Performans sınıf: fren disk/balata, lastik seti, soğutma ve turbo/boost kaçak kontrolü yaptır.")
        checklist.insert(1, "OBD/diagnostic: misfire, knock, sıcaklık değerleri ve şanzıman adaptasyon kayıtlarına baktır.")

    for item in base_list:
        if len(checklist) >= max_items:
            break
        if item not in checklist:
            checklist.append(item)

    return checklist[:max_items]


# =========================================================
# PREMIUM TEMPLATE
# =========================================================
def build_premium_template(req: AnalyzeRequest, enriched: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic 'Detaylı Analiz' çıktısı.
    - Şemayı korur (Flutter/UI kırılmasın)
    - Kullanıcı profiline (şehir, km, kullanım, yakıt/vites tercihleri) göre metni değiştirir
    - Araç yakıt/vites türüne uymayan kronik/risk maddelerini üretmez (DPF/EGR sadece dizel gibi)
    """
    v = req.vehicle
    p = req.profile or Profile()

    # -------------------------
    # Core data
    # -------------------------
    title = f"{v.year or ''} {v.make} {v.model}".strip() or "Detaylı Analiz"

    costs = enriched.get("costs") or {}
    ins = enriched.get("insurance", {}) or {}
    taxes = enriched.get("taxes", {}) or {}
    fixed = enriched.get("fixed_costs", {}) or {}
    risk = enriched.get("risk") or {}
    market = enriched.get("market", {}) or {}
    idx = (market.get("indices") or {}) if isinstance(market, dict) else {}

    listed = costs.get("listed_price_try")

    maint_min = int(costs.get("maintenance_yearly_try_min") or 0)
    maint_max = int(costs.get("maintenance_yearly_try_max") or 0)
    maint_mid = int((maint_min + maint_max) / 2) if (maint_min and maint_max) else int(costs.get("maintenance_yearly_try_mid") or 0)

    fuel_mid = costs.get("yearly_fuel_tr_mid") or 0
    total_mid = int(maint_mid + (fuel_mid or 0))

    # safety fallbacks
    if maint_mid <= 0:
        maint_mid = int(max(0, costs.get("maintenance_routine_yearly_est") or 0) + max(0, costs.get("maintenance_risk_reserve_yearly_est") or 0))
    if total_mid <= 0:
        total_mid = int(maint_mid + (fuel_mid or 0))

    # band helpers
    maint_band = (maint_min, maint_max) if (maint_min and maint_max) else (max(0, int(maint_mid * 0.85)), int(maint_mid * 1.15))
    total_band = (max(0, int(total_mid * 0.9)), int(total_mid * 1.1))

    # -------------------------
    # Vehicle/Profile signals
    # -------------------------
    fuel_s = _norm(v.fuel or p.fuel_preference or "")
    trans_s = _norm(v.transmission or p.transmission_preference or "")
    usage_s = _norm(p.usage or "mixed")
    city_name = (p.city or "").strip() or None

    is_diesel = any(k in fuel_s for k in ["dizel", "diesel"])
    is_lpg = "lpg" in fuel_s
    is_electric = any(k in fuel_s for k in ["elektrik", "electric", "ev"])
    is_hybrid = "hybrid" in fuel_s or "hibrit" in fuel_s

    is_manual = any(k in trans_s for k in ["manuel", "manual"])
    is_auto = any(k in trans_s for k in ["otomatik", "automatic", "auto", "dct", "cvt", "edc", "tiptronic"])

    yearly_km = int(getattr(p, "yearly_km", 0) or 0)

    # congestion context (optional)
    cong = get_city_congestion_context(req)
    cong_note = None
    try:
        if cong.get("ok") and isinstance(cong.get("row"), dict):
            row = cong.get("row") or {}
            lvl = row.get("congestion_level_tr") or row.get("congestion_level") or None
            extra = row.get("extra_note_tr") or None
            # short, useful, non-fake line
            if lvl:
                cong_note = f"{(cong.get('city_name') or city_name or 'Şehir')} merkez trafiği: {lvl}" + (f" ({extra})" if extra else "")
    except Exception:
        cong_note = None

    # -------------------------
    # Risk -> score
    # -------------------------
    base_risk = str(risk.get("baseline_risk_level") or "orta").strip().lower()
    if base_risk not in ("düşük", "orta", "orta-yüksek", "yüksek"):
        base_risk = "orta"

    overall = _score_from_risk(base_risk)

    # Uncertainty nudge
    uncertainty = build_uncertainty(enriched)
    if (uncertainty or {}).get("level") == "yüksek":
        overall = _clamp(overall - 5, 0, 100)
    elif (uncertainty or {}).get("level") == "orta":
        overall = _clamp(overall - 2, 0, 100)

    # -------------------------
    # Indices -> sub scores (keep existing fields)
    # -------------------------
    parts_av = int(idx.get("parts_availability_1_5") or 3)
    service_av = int(idx.get("service_network_1_5") or 3)
    resale_speed = int(idx.get("resale_speed_1_5") or 3)
    reliability = int(idx.get("reliability_1_5") or 3)

    # Convert to 0-100
    parts_100 = _to_100_from_1_5(parts_av)
    service_100 = _to_100_from_1_5(service_av)
    resale_100 = _to_100_from_1_5(resale_speed)
    reliability_100 = _to_100_from_1_5(reliability)

    # These map to the cards you're already using
    economy_100 = _clamp(int(0.55 * (100 - _clamp(int(total_mid / 3000), 0, 100)) + 0.45 * reliability_100), 35, 95)
    comfort_100 = _clamp(int(0.55 * overall + 0.45 * (service_100)), 40, 95)
    family_use_100 = _clamp(int(0.50 * overall + 0.50 * resale_100), 35, 95)
    electronics_100 = _clamp(int(0.60 * overall + 0.40 * reliability_100), 35, 95)

    # Personal fit (profile-first)
    # - city traffic penalizes manual in heavy city usage
    # - yearly_km high favors diesel (if diesel) else favors efficient fuel choice
    personal_fit = 72
    # usage weight
    if usage_s in ("city", "şehir", "sehir"):
        personal_fit += 2
        if is_manual and (city_name or cong_note):
            personal_fit -= 6
    elif usage_s in ("highway", "uzun", "otoyol"):
        personal_fit += 1
    else:
        personal_fit += 0

    if yearly_km >= 30000:
        personal_fit += (4 if is_diesel or is_hybrid or is_electric else 1)
    elif yearly_km <= 8000:
        personal_fit += (3 if (not is_diesel) else 0)

    # preference mismatch penalty
    pref = _norm(p.transmission_preference or "any")
    if pref in ("auto", "automatic", "otomatik") and is_manual:
        personal_fit -= 5
    if pref in ("manual", "manuel") and is_auto:
        personal_fit -= 4

    personal_fit_score = _clamp(int(personal_fit), 0, 100)

    # -------------------------
    # Smart, non-generic reasoning snippets ("zeka" dağıtılmış)
    # -------------------------
    def _one_line(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip())

    triggers: List[str] = []
    if is_diesel and usage_s in ("city", "şehir", "sehir"):
        triggers.append("Dizel + ağırlıkla şehir içi kullanım: DPF/EGR tarafı daha hassas (kısa mesafe/stop&go).")

    if is_manual and (city_name or cong_note) and usage_s in ("city", "şehir", "sehir", "mixed", "karma"):
        triggers.append("Şehir trafiği + manuel vites: konfor ve yorgunluk maliyeti artabilir (özellikle dur-kalk).")

    if yearly_km >= 30000:
        triggers.append("Yıllık km yüksek: yakıt ekonomisi ve bakım disiplininin önemi yükselir (maliyet sapması büyür).")

    # remove contradictions
    triggers = [_one_line(t) for t in triggers if t]

    # -------------------------
    # Risk items (vehicle-specific, rule-like)
    # -------------------------
    risk_lines: List[str] = []
    # DPF/EGR only diesel
    if is_diesel:
        if usage_s in ("city", "şehir", "sehir"):
            risk_lines.append("DPF / EGR: **Orta** — şehir içi kısa mesafede dolma/kurum riski artar; ara ara uzun yol rejenerasyon iyi gelir.")
        else:
            risk_lines.append("DPF / EGR: **Düşük-Orta** — kullanım karma/uzun yola yakınsa risk daha kontrollü.")
    # fuel system
    if v.mileage_km and v.mileage_km >= 120000:
        risk_lines.append("Yakıt sistemi: **Orta** — km bandı yükseldikçe enjektör/yakıt hattı hassasiyeti artabilir; düzgün bakım kayıtları puanı yukarı taşır.")
    else:
        risk_lines.append("Yakıt sistemi: **Düşük-Orta** — mevcut km bandında temel risk, bakım geçmişi ve yakıt kalitesiyle yönetilir.")

    # transmission
    if is_auto:
        risk_lines.append("Şanzıman: **Düşük-Orta** — otomatiklerde yağ/bakım geçmişi kritik; test sürüşünde vuruntu/gecikme aranmalı.")
    elif is_manual:
        risk_lines.append("Şanzıman: **Düşük** — manuel yapıda ana risk debriyaj/volan yıpranması; test sürüşüyle anlaşılır.")

    # -------------------------
    # Parts/Service/Market block (keep as scored table, with 'why')
    # -------------------------
    def _why_1_5(x: int, label: str) -> str:
        x = _clamp(int(x), 1, 5)
        if x >= 4:
            return f"{label} güçlü: yaygınlık ve alternatif çok."
        if x == 3:
            return f"{label} ortalama: bulunur ama fiyatta oynama olabilir."
        return f"{label} zayıf: doğru usta/parça için arama süresi uzayabilir."

    parts_why = _why_1_5(parts_av, "Parça erişimi")
    service_why = _why_1_5(service_av, "Servis/usta ağı")
    resale_why = _why_1_5(resale_speed, "2. el likidite")
    reliab_why = _why_1_5(reliability, "Dayanıklılık algısı")

    # -------------------------
    # Cost table
    # -------------------------
    mtv = (taxes.get("mtv") or {})
    insp = (fixed.get("inspection") or {})
    traffic = (ins.get("traffic") or {})
    kasko = (ins.get("kasko") or {})

    # -------------------------
    # Build text (same overall format you like)
    # -------------------------
    short_comment = ""
    if personal_fit_score >= 80 and overall >= 70:
        short_comment = "Profilinle uyumu yüksek; doğru kontrolle alınabilir bir ilan."
    elif overall >= 70:
        short_comment = "Genel tablo fena değil; kararın expertiz + geçmiş kontrolüyle netleşir."
    else:
        short_comment = "Fiyat/koşul doğruysa alınabilir; ama risk alanları netleştirilmeden acele etme."

    # Pros/cons: keep short and non-generic
    pros: List[str] = []
    cons: List[str] = []

    if parts_av >= 4:
        pros.append("Parça/servis erişimi rahat; kullanım-satış kolaylığı sağlar.")
    if resale_speed >= 4:
        pros.append("2. el hareketli; doğru fiyatla satış süresi genelde kısa olur.")
    if is_diesel and yearly_km >= 20000:
        pros.append("Yüksek km için dizel ekonomi avantajı mantıklı (bakım disiplinin varsa).")

    if is_diesel and usage_s in ("city", "şehir", "sehir"):
        cons.append("Şehir içi ağırlık varsa DPF/EGR tarafını izlemek gerekir.")
    if is_manual and (city_name or cong_note) and usage_s in ("city", "şehir", "sehir", "mixed", "karma") and (p.transmission_preference or "any") not in ("manual", "manuel"):
        cons.append("Trafikte manuel konforu düşürür; otomatik isteyen kullanıcıyı yorabilir.")
    if (uncertainty or {}).get("level") in ("orta", "yüksek"):
        cons.append("Belirsizlik seviyesi yüksek; tramer/bakım geçmişi skoru ciddi değiştirir.")

    if not pros:
        pros = ["Segmentine göre dengeli bir ilan; doğru kontrolle netleşir."]
    if not cons:
        cons = ["Kritik nokta: geçmiş kayıtlar ve expertiz sonucu."]
    pros = pros[:4]
    cons = cons[:4]

    # Critical points (for UI)
    critical_points: List[str] = []
    if is_diesel and usage_s in ("city", "şehir", "sehir"):
        critical_points.append("DPF/EGR durumunu sor (rejenerasyon, uyarı ışığı, kullanım tipi).")

    if is_auto:
        critical_points.append("Şanzıman bakım/yağ geçmişi + test sürüşünde vuruntu/gecikme kontrolü.")
    else:
        critical_points.append("Debriyaj/volan yıpranma belirtileri (kavrama, titreme) kontrolü.")

    critical_points.append("Tramer + bakım kayıtları + OBD taraması (belirsizliği düşürür).")

    # Build section strings
    # 0) Snapshot
    snapshot_lines = [
        f"GENEL SKOR: {overall} / 100",
        f"GENEL RİSK SEVİYESİ: {('Düşük' if base_risk=='düşük' else 'Orta' if base_risk=='orta' else 'Orta-Yüksek' if base_risk=='orta-yüksek' else 'Yüksek')}",
        f"BELİRSİZLİK: {str((uncertainty or {}).get('level') or 'orta').title()} ({int((uncertainty or {}).get('score_0_100') or 0)} / 100)",
        f"YILLIK TOPLAM MALİYET (ORTA): ~{_fmt_try(total_mid)}",
        f"KİŞİSEL UYGUNLUK: {('Yüksek' if personal_fit_score>=80 else 'Orta' if personal_fit_score>=60 else 'Düşük')}",
    ]

    # 1) Scores + explanations (mechanical only, but keep others)
    sec1 = []
    sec1.append("1) Genel Teknik & Kullanım Değerlendirmesi (puanlı)")
    sec1.append("")
    sec1.append(f"Mekanik: { _clamp(overall+2,0,100) } / 100")
    sec1.append(f"Ekonomi: { economy_100 } / 100")
    sec1.append(f"Konfor: { comfort_100 } / 100")
    sec1.append(f"2. El: { resale_100 } / 100")
    sec1.append(f"Uygunluk: { personal_fit_score } / 100")
    sec1.append("")
    # distributed intelligence here:
    mech_expl = []
    mech_expl.append("**Mekanik:**")
    mech_expl.append("Bu skor, motor/aktarma + kilometre bandı + bakım disiplini varsayımıyla hesaplanır.")
    if v.mileage_km:
        mech_expl.append(f"Mevcut kilometre: **{_fmt_int(v.mileage_km)} km** — bu bantta kararın kaderi *geçmiş kayıtlar* ve *expertiz* olur.")
    if triggers:
        mech_expl.append("Bu araç-profil kombinasyonunda öne çıkan tetikleyiciler:")
        for t in triggers[:3]:
            mech_expl.append(f"- {t}")
    sec1.append("\n".join(mech_expl))

    eco_expl = []
    eco_expl.append("\n**Ekonomi:**")
    eco_expl.append(f"Yıllık toplam maliyeti belirleyen iki ana parça var: **bakım/rezerv** + **yakıt**. Bu ilanda orta senaryoda ~{_fmt_try(total_mid)} çıkıyor.")
    if yearly_km:
        eco_expl.append(f"Yıllık km (**{_fmt_int(yearly_km)}**) arttıkça yakıt payı büyür; küçük tüketim farkı bile yıllık tabloda hissedilir.")
    sec1.append("\n".join(eco_expl))

    kon_expl = []
    kon_expl.append("\n**Konfor:**")
    if is_manual and (city_name or cong_note):
        kon_expl.append("Manuel vites, yoğun trafikte konforu düşürür; uzun yolda ise sorun olmaz.")
    else:
        kon_expl.append("Konfor skorunu; kullanım tipi, servis/usta erişimi ve genel risk seviyesi birlikte belirliyor.")
    if cong_note:
        kon_expl.append(f"- {cong_note}")
    sec1.append("\n".join(kon_expl))

    resale_expl = []
    resale_expl.append("\n**2. El:**")
    resale_expl.append("Bu puan; modelin piyasadaki satılabilirliği + ilan yoğunluğu + doğru fiyatlamaya duyarlılığa göre oluşur.")
    sec1.append("\n".join(resale_expl))

    fit_expl = []
    fit_expl.append("\n**Uygunluk (profil bazlı):**")
    fit_expl.append(f"Kullanım: **{p.usage}**, Yakıt tercihi: **{p.fuel_preference}**, Vites tercihi: **{p.transmission_preference or 'any'}**.")
    if city_name:
        fit_expl.append(f"Şehir: **{city_name}**.")
    if is_manual and (city_name or cong_note) and (p.transmission_preference or "any") not in ("manual", "manuel"):
        fit_expl.append("Bu profilde manuel tercih, *şehir içi yorgunluk maliyeti* yüzünden uygunluğu aşağı çekiyor.")
    if is_diesel and usage_s in ("city", "şehir", "sehir"):
        fit_expl.append("Dizel seçimi yakıtta avantajlı; fakat şehir içi ağırlık varsa DPF/EGR tarafını bilinçli kullanmak gerekir.")
    sec1.append("\n".join(fit_expl))

    # 2) Cost
    sec2 = []
    sec2.append("\n2) Yıllık Maliyet Özeti (tahmini)")
    sec2.append(f"TOPLAM (bakım + yakıt, orta senaryo): **~{_fmt_try(total_mid)} / yıl**")
    sec2.append("")
    sec2.append("Kalem\tYıllık Tahmin")
    # Use existing band formatters if present, else fallback
    def _band(a, b):
        try:
            return _fmt_band_try(int(a), int(b))
        except Exception:
            return f"{_fmt_try(int(a))} – {_fmt_try(int(b))}"

    maint_routine = int(costs.get("maintenance_routine_yearly_est") or int(maint_mid * 0.65))
    maint_reserve = int(costs.get("maintenance_risk_reserve_yearly_est") or max(0, maint_mid - maint_routine))

    sec2.append(f"Rutin + olası bakım\t{_band(max(0, maint_routine), max(maint_routine, maint_mid))}")
    sec2.append(f"Yakıt\t~{_fmt_try(int(fuel_mid))}")
    # taxes/ins bands (safe)
    try:
        mtv_min = int((mtv.get("min") or mtv.get("mid") or 0))
        mtv_max = int((mtv.get("max") or mtv.get("mid") or mtv_min))
        if mtv_min and mtv_max:
            sec2.append(f"MTV\t{_band(mtv_min, mtv_max)}")
    except Exception:
        pass
    try:
        insp_mid = int((insp.get("mid") or 0))
        if insp_mid:
            sec2.append(f"Muayene\t~{_fmt_try(insp_mid)}")
    except Exception:
        pass
    try:
        tr_min = int((traffic.get("min") or 0))
        tr_max = int((traffic.get("max") or tr_min))
        if tr_min:
            sec2.append(f"Trafik sigortası\t{_band(tr_min, tr_max)}")
    except Exception:
        pass
    try:
        k_min = int((kasko.get("min") or 0))
        k_max = int((kasko.get("max") or k_min))
        if k_min:
            sec2.append(f"Kasko\t{_band(k_min, k_max)}")
    except Exception:
        pass

    sec2.append("")
    sec2.append(f"Bu band; **{_fmt_int(yearly_km) if yearly_km else 'varsayılan'} km/yıl**, kullanım: **{p.usage}** varsayımıyla hesaplanır.")

    # 3) Risk
    sec3 = []
    sec3.append("\n3) Risk Profili (araç özel)\n")
    sec3.append(f"Genel risk seviyesi: **{('Düşük' if base_risk=='düşük' else 'Orta' if base_risk=='orta' else 'Orta-Yüksek' if base_risk=='orta-yüksek' else 'Yüksek')}**")
    sec3.append("Bu değerlendirme; **bu motor + bu kilometre + bu kullanım profili** için geçerlidir.")
    sec3.append("")
    for rl in risk_lines[:4]:
        sec3.append(f"- {rl}")

    # 4) Parts/service & market as scored table
    sec4 = []
    sec4.append("\n4) Parça, Servis & Piyasa Durumu")
    sec4.append("Alan\tPuan\tAçıklama")
    sec4.append(f"Parça bulunabilirliği\t{parts_av} / 5\t{parts_why}")
    sec4.append(f"Servis / usta ağı\t{service_av} / 5\t{service_why}")
    sec4.append(f"2. el likidite\t{resale_speed} / 5\t{resale_why}")
    sec4.append(f"Dayanıklılık algısı\t{reliability} / 5\t{reliab_why}")
    sec4.append("\nNot: Bu puanlar model/segment verisi + Türkiye piyasası genel davranışına göre kalibre edilir; ilan özelindeki durum expertizle netleşir.")

    # 5) Personal fit (expanded)
    sec5 = []
    sec5.append("\n5) Kişiye Uygunluk (profil bazlı)")
    sec5.append("")
    sec5.append("Kullanıcı profili:")
    sec5.append(f"- Yıllık km: **{_fmt_int(yearly_km)}**")
    if city_name:
        sec5.append(f"- Şehir: **{city_name}**")
    sec5.append(f"- Kullanım: **{p.usage}**")
    sec5.append(f"- Yakıt tercihi: **{p.fuel_preference}**")
    sec5.append(f"- Vites tercihi: **{p.transmission_preference or 'any'}**")
    sec5.append("")
    sec5.append(f"Uygunluk değerlendirmesi: **{('Yüksek' if personal_fit_score>=80 else 'Orta' if personal_fit_score>=60 else 'Düşük')}** ({personal_fit_score}/100)")
    sec5.append("")
    # The "smart" part: conditional, concrete, not generic
    if city_name and city_name.lower().startswith("istan"):
        if is_manual and yearly_km >= 20000 and usage_s in ("city", "şehir", "sehir", "mixed", "karma"):
            sec5.append("İstanbul senaryosunda (dur-kalk + yoğun trafik), **manuel vites** uzun vadede yorar; otomatik tercih eden kullanıcı için uyum puanı düşer.")
        elif is_auto and usage_s in ("city", "şehir", "sehir"):
            sec5.append("İstanbul gibi dur-kalk yoğun şehirlerde **otomatik vites**, konfor ve kullanım sürdürülebilirliği açısından avantajlıdır.")
    if cong_note:
        sec5.append(f"Trafik bağlamı: {cong_note}.")
    if is_diesel and usage_s in ("city", "şehir", "sehir"):
        sec5.append("Dizel tercihinde karar: yakıt ekonomisi kazanırsın; karşılığında **kısa mesafe kullanımını yönetmen** gerekir (ara ara uzun yol, doğru yağ, doğru yakıt).")

    # 6) Uncertainty
    sec6 = []
    sec6.append("\n6) Belirsizlik & Netleştirme")
    sec6.append(f"Belirsizlik seviyesi: **{str((uncertainty or {}).get('level') or 'orta').title()}** ({int((uncertainty or {}).get('score_0_100') or 0)}/100)")
    sec6.append("Belirsizliği en hızlı düşüren kontroller:")
    sec6.append("- Tramer kaydı + değişen/boya bilgisi")
    sec6.append("- Servis/bakım geçmişi (fatura/kayıt)")
    sec6.append("- OBD taraması + test sürüşü (özellikle kritik parçalar)")
    if is_diesel:
        sec6.append("- DPF/EGR geçmişi (rejenerasyon/temizlik)")
    if is_auto:
        sec6.append("- Şanzıman bakım/yağ geçmişi")

    # 7) Checklist
    sec7 = []
    sec7.append("\n7) Satın Alma Öncesi Kontrol Listesi")
    for cp in critical_points[:5]:
        sec7.append(f"- {cp}")

    # 8) Final note
    sec8 = []
    sec8.append("\n8) Son Karar Notu")
    sec8.append(_one_line(short_comment + " " + "Karar, fiyat + expertiz + geçmiş kayıtların netliği ile kesinleşir."))

    # Compose cards and new result text (card-based format)
    # Extract segment and info level for display
    seg = enriched.get("segment") or {}
    seg_name = None
    if isinstance(seg, dict):
        seg_name = seg.get("name_tr") or seg.get("name")
    info_q = enriched.get("info_quality") or {}
    level = None
    if isinstance(info_q, dict):
        level = info_q.get("level_tr") or info_q.get("level")
    # Translation helpers for usage, transmission and fuel
    def tr_usage(u: str) -> str:
        mapping = {
            "mixed": "Karma kullanım",
            "karma": "Karma kullanım",
            "city": "Şehir içi",
            "şehir": "Şehir içi",
            "sehir": "Şehir içi",
            "highway": "Uzun yol",
            "uzun": "Uzun yol",
            "otoyol": "Uzun yol",
        }
        return mapping.get((u or "").lower(), u or "Belirtilmemiş")
    def tr_trans(t: str) -> str:
        mapping = {
            "auto": "Otomatik",
            "automatic": "Otomatik",
            "otomatik": "Otomatik",
            "manual": "Manuel",
            "manuel": "Manuel",
            "any": "Fark etmez",
            "": "Fark etmez",
        }
        return mapping.get((t or "").lower(), t or "Belirtilmemiş")
    def tr_fuel(f: str) -> str:
        mapping = {
            "diesel": "Dizel",
            "dizel": "Dizel",
            "gasoline": "Benzin",
            "benzin": "Benzin",
            "lpg": "LPG",
            "hybrid": "Hibrit",
            "hibrit": "Hibrit",
            "electric": "Elektrik",
            "ev": "Elektrik",
        }
        return mapping.get((f or "").lower(), f or "Belirtilmemiş")
    # Labels for risk and personal fit
    risk_label = "Düşük" if base_risk == "düşük" else ("Orta" if base_risk == "orta" else ("Orta-Yüksek" if base_risk == "orta-yüksek" else "Yüksek"))
    personal_label = "Yüksek" if personal_fit_score >= 80 else ("Orta" if personal_fit_score >= 60 else "Düşük")
    cards = []

    # Safe labels
    unc_level = str((uncertainty or {}).get("level") or "orta").strip()
    try:
        unc_score = int((uncertainty or {}).get("score_0_100") or 0)
    except Exception:
        unc_score = 0

    # -------------------------
    # Card 0 — Özet (Tek Bakış)
    # -------------------------
    c0 = []
    c0.append("**Premium Araç Analizi**")
    c0.append(f"**Araç:** {title}")
    if v.mileage_km is not None:
        c0.append(f"**Kilometre:** {_fmt_int(int(v.mileage_km))} km")
    if v.fuel:
        c0.append(f"**Yakıt:** {tr_fuel(v.fuel)}")
    if v.transmission:
        c0.append(f"**Vites:** {tr_trans(v.transmission)}")
    if seg_name:
        c0.append(f"**Segment:** {seg_name}")
    if level:
        c0.append(f"**Bilgi Seviyesi:** {level}")
    c0.append("")
    c0.append("### Genel Karar Özeti")
    c0.append(f"- **GENEL SKOR:** {overall} / 100")
    c0.append(f"- **GENEL RİSK SEVİYESİ:** {risk_label}")
    c0.append(f"- **BELİRSİZLİK:** {unc_level.capitalize()} ({unc_score} / 100)")
    c0.append(f"- **YILLIK TOPLAM MALİYET (ORTA):** ~{_fmt_try(total_mid)}")
    c0.append(f"- **KİŞİSEL UYGUNLUK:** {personal_label}")
    c0.append("")
    c0.append(_one_line(short_comment))
    cards.append({"title": "⭐ Özet (Tek Bakış)", "content": "\n".join(c0)})

    # -------------------------
    # Card 1 — Karar Mantığı
    # -------------------------
    c1 = []
    c1.append("Bu analiz; araç + ilan bilgisi + segment verisi + kullanıcı profili birlikte düşünülerek hazırlanır.")
    c1.append("")
    c1.append("**Bu araç ‘alınabilir’ tarafa geçer, eğer:**")
    c1.append("- Bakım geçmişi net (fatura/servis kaydı, düzenli periyot).")
    if is_auto:
        c1.append("- Test sürüşünde **ısınınca** gecikme/vuruntu/sarsıntı yok (şanzıman davranışı pürüzsüz).")
    else:
        c1.append("- Test sürüşünde çekiş/tekleme/titreşim yok; kavrama/aktarma düzgün.")
    c1.append("- Tramer + ekspertiz + OBD taraması temiz (belirsizlik hızla düşer).")
    c1.append("")
    c1.append("Bu üçlü temizse risk bandı düşer, yıllık maliyet tahmini daralır ve pazarlık payın artar.")
    cards.append({"title": "🧠 Karar Mantığı", "content": "\n".join(c1)})

    # -------------------------
    # Card 2 — Genel Teknik & Kullanım (Puanlı)
    # -------------------------
    c2 = []
    c2.append(f"**Mekanik:** {_clamp(overall + 2, 0, 100)} / 100")
    c2.append(f"**Ekonomi:** {economy_100} / 100")
    c2.append(f"**Konfor:** {comfort_100} / 100")
    c2.append(f"**2. El:** {resale_100} / 100")
    c2.append(f"**Kişisel Uygunluk:** {personal_fit_score} / 100")
    c2.append("")
    c2.append("**Bu skoru yükselten faktörler:**")
    c2.append("- Segmentine göre dengeli karakter + yönetilebilir bakım bandı.")
    c2.append("- Parça/usta erişimi iyi (doğru yer seçilirse).")
    if yearly_km >= 25000:
        c2.append("- Yüksek km/yıl kullanımında doğru yakıt tercihi maliyeti aşağı çeker.")
    c2.append("")
    c2.append("**Bu skoru sınırlayan faktörler:**")
    if triggers:
        for t in triggers[:3]:
            c2.append(f"- {t}")
    else:
        c2.append("- Kritik nokta: geçmiş kayıtlar + ekspertiz sonucu (belirsizlik).")
    if is_diesel and usage_s in ("city", "şehir", "sehir"):
        c2.append("- Şehir içi yoğun kullanım dizelde kurum/DPF/EGR yükünü artırabilir.")
    cards.append({"title": "⚙️ Genel Teknik & Kullanım (Puanlı)", "content": "\n".join(c2)})

    # -------------------------
    # Card 3 — Yıllık Maliyet (Detaylı)
    # -------------------------
    c3 = []
    fuel_min = costs.get("yearly_fuel_tr_min")
    fuel_max = costs.get("yearly_fuel_tr_max")
    fuel_mid = costs.get("yearly_fuel_tr_mid")

    _fuel_min_eff = int(fuel_min or fuel_mid or 0)
    _fuel_max_eff = int(fuel_max or fuel_mid or 0)
    _maint_min_eff = int(maint_min or maint_mid or 0)
    _maint_max_eff = int(maint_max or maint_mid or 0)

    op_min = int(_maint_min_eff + _fuel_min_eff)
    op_max = int(_maint_max_eff + _fuel_max_eff)

    c3.append("**Yakıt + bakım toplamı (tahmini):**")
    c3.append(f"- Alt/Üst: ~{_fmt_try(op_min)} – ~{_fmt_try(op_max)} / yıl")
    c3.append(f"- Orta senaryo: ~{_fmt_try(total_mid)} / yıl")
    c3.append("")

    if fuel_mid is not None or fuel_min is not None:
        if fuel_min is not None and fuel_max is not None:
            c3.append(f"**Yakıt:** ~{_fmt_try(fuel_min)} – ~{_fmt_try(fuel_max)} (orta ~{_fmt_try(fuel_mid)})")
        else:
            c3.append(f"**Yakıt:** orta ~{_fmt_try(fuel_mid)}")

    routine = costs.get("maintenance_routine_yearly_est")
    reserve = costs.get("maintenance_risk_reserve_yearly_est")
    if routine is not None:
        c3.append(f"**Rutin bakım:** ~{_fmt_try(routine)} / yıl")
    if reserve is not None:
        c3.append(f"**Olası bakım/rezerv:** ~{_fmt_try(reserve)} / yıl")

    c3.append("")
    c3.append("**Diğer yıllık kalemler (kişiye göre oynar):**")
    traffic = (ins or {}).get("traffic") or {}
    kasko = (ins or {}).get("kasko") or {}

    if isinstance(traffic, dict) and traffic.get("ok"):
        c3.append(
            f"- Trafik sigortası (band): ~{_fmt_try(traffic.get('traffic_est_try_min'))}"
            f" – ~{_fmt_try(traffic.get('traffic_est_try_max'))} (orta ~{_fmt_try(traffic.get('traffic_est_try_mid'))})"
        )
    else:
        c3.append("- Trafik sigortası: profil/veri eksik → band hesaplanamadı.")

    if isinstance(kasko, dict) and kasko.get("ok"):
        c3.append(
            f"- Kasko (band): ~{_fmt_try(kasko.get('kasko_try_min'))}"
            f" – ~{_fmt_try(kasko.get('kasko_try_max'))} (orta ~{_fmt_try(kasko.get('kasko_try_mid'))})"
        )
    else:
        c3.append("- Kasko: fiyat/profil/veri eksik → band hesaplanamadı.")

    mtv = ((taxes or {}).get("mtv") or {})
    if mtv.get("ok"):
        c3.append(f"- MTV (yıllık): ~{_fmt_try(mtv.get('mtv_yearly_try_mid'))} (band ~{_fmt_try(mtv.get('mtv_yearly_try_min'))}–{_fmt_try(mtv.get('mtv_yearly_try_max'))})")

    insp = ((fixed or {}).get("inspection") or {})
    if insp.get("ok"):
        c3.append(f"- Muayene (yıllık ort.): ~{_fmt_try(insp.get('inspection_yearly_avg_try'))}")

    c3.append("")
    c3.append("**Maliyeti en çok oynatan 3 değişken:**")
    c3.append("- Sigorta/kasko basamağı + il/hasar geçmişi")
    c3.append("- Bakım disiplini (zamanında yağ/filtre + doğru işçilik)")
    c3.append("- Kullanım oranı (şehir içi/uzun yol) ve trafik yoğunluğu")

    cards.append({"title": "💰 Yıllık Maliyet (Detaylı)", "content": "\n".join(c3)})

# -------------------------
    # Card 4 — Risk Profili
    # -------------------------
    c4 = []
    c4.append(f"**Genel risk seviyesi:** {risk_label}")
    c4.append("Bu risk; 'kesin arıza var' demek değil, kontrol edilmezse masraf çıkarma ihtimali var demektir.")
    c4.append("")
    if is_diesel:
        if usage_s in ("city", "şehir", "sehir"):
            c4.append("**DPF/EGR:** Orta — şehir içi kısa mesafe ağırlığında kurum döngüsü artar; geçmiş/uyarı ışığı sorgulanmalı.")
        else:
            c4.append("**DPF/EGR:** Düşük-Orta — uzun yol payı varsa daha yönetilebilir; yine de geçmiş ve uyarı kaydı kontrol edilmeli.")
    else:
        c4.append("**Yakıt sistemi:** Düşük-Orta — bu km/yaş bandında temel risk, bakım geçmişi ve yakıt kalitesiyle yönetilir.")
    if is_auto:
        c4.append("**Şanzıman:** Düşük-Orta — otomatikte bakım/yağ geçmişi kritik; test sürüşünde vuruntu/gecikme aranmalı.")
    elif is_manual:
        c4.append("**Şanzıman:** Düşük — manuel yapıda ana risk debriyaj/volan yıpranması; test sürüşüyle anlaşılır.")
    c4.append("**Genel yıpranma:** Normal — alt takım/lastik/fren gibi kalemler km bandında beklenen masraflardır.")
    c4.append("")
    c4.append("**Hızlı kontrol ipuçları:**")
    c4.append("- Soğuk/ılık çalıştırma + rölanti stabil mi?")
    c4.append("- Test sürüşünde çekiş dalgalanması / tekleme / titreme var mı?")
    c4.append("- OBD taramasında sürekli hata/tekrarlayan uyarı var mı?")
    cards.append({"title": "⚠️ Risk Profili", "content": "\n".join(c4)})

    # -------------------------
    # Card 5 — Parça / Servis & Piyasa
    # -------------------------
    c5 = []
    c5.append(f"**Parça bulunabilirliği:** {parts_av} / 5")
    c5.append(f"**Servis/usta ağı:** {service_av} / 5")
    c5.append(f"**Piyasa hızı:** {resale_speed} / 5")
    c5.append("")
    c5.append("**Neye göre? (kısa)**")
    c5.append(f"- {parts_why}")
    c5.append(f"- {service_why}")
    c5.append(f"- {resale_why}")
    c5.append("")
    c5.append("**Pratik not:**")
    c5.append("- Parça/usta tarafı iyi olsa bile, ilan özelindeki durum (bakım/hasar) fiyat pazarlığında daha belirleyicidir.")
    cards.append({"title": "🧩 Parça / Servis & Piyasa", "content": "\n".join(c5)})

# -------------------------
    # Card 6 — Kişiye Uygunluk
    # -------------------------
    c6 = []
    c6.append("**Profil:**")
    c6.append(f"- Yıllık km: {_fmt_int(yearly_km)}")
    c6.append(f"- Kullanım: {('Şehir içi' if usage_s in ('city','şehir','sehir') else 'Uzun yol' if usage_s=='highway' else 'Karışık')}")
    if city_name:
        c6.append(f"- Şehir: {city_name}")
    if p.fuel_preference:
        c6.append(f"- Yakıt tercihi: {tr_fuel(p.fuel_preference)}")
    if p.transmission_preference:
        c6.append(f"- Vites tercihi: {tr_trans(p.transmission_preference)}")
    c6.append("")
    c6.append(f"**Uygunluk:** {personal_fit_score}/100 ({personal_label})")
    if yearly_km >= 25000:
        c6.append("- Yüksek km/yıl kullanımında yakıt ve bakım disiplini belirleyici; doğru tercih toplam maliyeti ciddi etkiler.")
    if city_name or cong_note:
        c6.append(f"- Trafik etkisi: {cong_note or 'Şehir içi yoğunluk kullanım karakterini belirler.'}")
    c6.append("")
    c6.append("**Kime daha uygun?**")
    c6.append("- Dengeli kullanım isteyen (ekonomi + konfor) kullanıcılar.")
    if is_auto:
        c6.append("- Şehir trafiğinde otomatik rahatlığını isteyen kullanıcılar.")
    cards.append({"title": "👤 Kişiye Uygunluk", "content": "\n".join(c6)})

    # -------------------------
    # Card 7 — Kullanım Senaryosu Etkisi (mini)
    # -------------------------
    c7 = []
    if usage_s in ("city", "şehir", "sehir"):
        c7.append("- **Şehir içi ağırlık:** stop-go + kısa mesafe döngüsü kritik; bakım disiplininin önemi artar.")
        if is_diesel:
            c7.append("- Dizelde kısa mesafe yoğunluğu DPF/EGR yükünü artırabilir; uzun yol “kendini temizleme” fırsatı sağlar.")
    elif usage_s == "highway":
        c7.append("- **Uzun yol ağırlık:** tüketim daha stabil; termal denge daha iyi; risk daha kontrollü olur.")
        if is_diesel:
            c7.append("- Dizelde uzun yol payı DPF tarafını rahatlatır; yine de geçmiş kayıt sorgulanmalı.")
    else:
        c7.append("- **Karışık kullanım:** iyi denge; şehir içi oranı yükselirse bakım/risk payı büyür, uzun yol payı arttıkça risk azalır.")
        if is_diesel:
            c7.append("- Dizelde haftalık/aylık uzun yol payı DPF/EGR tarafında ciddi fark yaratır.")
    c7.append("- Bu yüzden en doğru karar: kullanım oranını (şehir içi % kaç?) netleştirip test sürüşünü ona göre yorumlamaktır.")
    cards.append({"title": "🛣️ Kullanım Senaryosu Etkisi", "content": "\n".join(c7)})

    # -------------------------
    # Card 8 — En Kritik 3 Soru
    # -------------------------
    q = []
    if is_diesel and usage_s in ("city", "şehir", "sehir"):
        q.append("DPF/EGR geçmişi var mı? (rejenerasyon, uyarı ışığı, kısa mesafe kullanımı)")
    if is_auto:
        q.append("Şanzıman bakım/yağ geçmişi net mi ve test sürüşünde vuruntu/gecikme var mı?")
    else:
        q.append("Debriyaj/volan yıpranma belirtisi var mı? (kavrama, titreme, ses)")
    q.append("Tramer + bakım kayıtları + OBD taraması temiz mi? (belirsizliği düşürür)")
    cards.append({"title": "❓ En Kritik 3 Soru", "content": "\n".join([f"{i+1}) {qq}" for i, qq in enumerate(q[:3])])})

    # -------------------------
    # Card 9 — Belirsizlik & Netleştirme
    # -------------------------
    c9 = []
    c9.append(f"**Belirsizlik seviyesi:** {unc_level.capitalize()} ({unc_score}/100)")
    c9.append("")
    c9.append("Belirsizliği en hızlı düşüren kontroller:")
    c9.append("- Tramer + boya/değişen bilgisi (kalem kalem)")
    c9.append("- Servis/bakım geçmişi (fatura/kayıt)")
    c9.append("- OBD taraması + test sürüşü (ısınınca davranış)")
    if is_diesel:
        c9.append("- Dizelde DPF/EGR davranışı ve uyarı geçmişi")
    cards.append({"title": "🔎 Belirsizlik & Netleştirme", "content": "\n".join(c9)})

    # -------------------------
    # Card 10 — Satın Alma Öncesi Checklist
    # -------------------------
    c10 = []
    c10.append("- Tramer + ekspertiz raporu (motor/alt takım/elektronik)")
    c10.append("- Test sürüşü (soğuk + ısınınca; çekiş/sarsıntı/titreşim kontrolü)")
    c10.append("- OBD taraması (sürekli hata, tekrarlayan uyarı)")
    if is_diesel:
        c10.append("- Dizelde DPF/EGR belirtileri (çekiş düşüşü, uyarı, duman)")
    if is_auto:
        c10.append("- Otomatikte geçiş pürüzsüz mü? (vuruntu/gecikme/sarsıntı)")
    else:
        c10.append("- Manuelde debriyaj kavrama/titreme/ses")
    cards.append({"title": "✅ Satın Alma Öncesi Checklist", "content": "\n".join(c10)})

    # -------------------------
    # Card 11 — Son Karar Notu
    # -------------------------
    c11 = []
    c11.append(_one_line(short_comment + " " + "Karar; fiyat + ekspertiz + geçmiş kayıtların netliği ile kesinleşir."))
    cards.append({"title": "🏁 Son Karar Notu", "content": "\n".join(c11)})

    # Build the new result text by joining cards
    result_text = "\n---\n".join([c['title'] + "\n\n" + c['content'] for c in cards]).strip()

    # -------------------------
    # Preview + price tag
    # -------------------------
    price_tag = None
    if isinstance(listed, (int, float)) and listed > 0:
        # very rough: compare listed to a mid guess if available
        mid_guess = costs.get("price_estimate_try_mid")
        try:
            mid_guess = int(mid_guess) if mid_guess else None
        except Exception:
            mid_guess = None
        if mid_guess:
            ratio = listed / max(1, mid_guess)
            if ratio <= 0.92:
                price_tag = "İyi"
            elif ratio <= 1.06:
                price_tag = "Normal"
            else:
                price_tag = "Yüksek"

    # score reason (for UI)
    score_reason = {
        "because": _one_line("; ".join(triggers[:2])) if triggers else "Profil + risk + maliyet dengesiyle skorlandı."
    }

    out = {
        "scores": {
            "overall_100": int(overall),
            "mechanical_100": int(_clamp(overall + 2, 0, 100)),
            "body_100": int(_clamp(overall - 1, 0, 100)),
            "economy_100": int(economy_100),
            "comfort_100": int(comfort_100),
            "family_use_100": int(family_use_100),
            "resale_100": int(resale_100),
            "electronics_100": int(electronics_100),
            "personal_fit_100": int(personal_fit_score),
        },
        "summary": {
            "short_comment": short_comment,
            "pros": pros,
            "cons": cons,
            "estimated_risk_level": base_risk,
        },
        "preview": {
            "title": title,
            "price_tag": price_tag,
            "spoiler": "Genel skor, yıllık toplam band, parça/servis geneli ve profil uyumu",
            "bullets": [
                "Genel skor + kısa gerekçe",
                "Yıllık toplam (bakım+yakıt) bandı",
                "Parça/servis + 2.el endeksi (neden?)",
                "Kişiye uygunluk (şehir/traffic dahil)",
            ],
        },
        "final_snapshot": {
            "score_because": score_reason.get("because"),
            "critical_points": critical_points[:3],
        },
        "result": result_text,
        # Include structured cards for client rendering
        "cards": cards,
    }
    return out
def premium_analyze_impl(req: AnalyzeRequest) -> Dict[str, Any]:
    enriched = build_enriched_context(req)
    base = build_premium_template(req, enriched)
    return base


# =========================================================
# LLM HELPERS (optional)
# =========================================================
def call_llm_json(model_name: str, system_prompt: str, user_content: str) -> Optional[Dict[str, Any]]:
    """
    OpenAI çağrısı (JSON mode). Önce OpenAI Python SDK (client) dener.
    Eğer SDK yoksa veya client None ise, HTTP (requests) fallback ile çağırır.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("LLM disabled: OPENAI_API_KEY missing")
        return None

    # 1) SDK yolu
    if client is not None:
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
            print("LLM error (sdk):", e)

    # 2) HTTP fallback yolu
    try:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model_name,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.7,
        }
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code != 200:
            print("LLM error (http) status=", r.status_code, "body=", (r.text or "")[:300])
            return None
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        return json.loads(content) if isinstance(content, str) else content
    except Exception as e:
        print("LLM error (http):", e)
        return None



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


SYSTEM_PROMPT_PREMIUM_ENRICH = """
Sen 'Oto Analiz' uygulaması için **Premium (Detaylı) analiz** çıktısını, kullanıcıya "yapay zeka konuşuyor" hissi verecek şekilde zenginleştiren yardımcı asistansın.

Girdi olarak sana:
- Araç bilgileri
- Kullanıcı profili
- Backend'in deterministik premium raporu (skorlar + kartlar + maliyet bandı vs.)

verilecek.

Görevin:
EK KURAL:
- **Uzunlatma.** Mevcut metni gereksiz uzatmak yasak; mümkünse kısalt, netleştir.
- En fazla **3-5 kritik nokta** seçip onları netleştir; diğer kısımları tek cümleyle toparla.
- Sert hüküm yok: "alınmaz/tehlikeli" gibi ifadeleri kullanma.
1) Mevcut raporu **bozmadan** üstüne "AI sesi" katmak (kısa, net, araç-özel).
2) Kullanıcı profilinden (şehir/kullanım/yıllık km/vites-yakıt tercihi/bütçe hassasiyeti) yola çıkarak **kişiye özel** 4-6 madde üretmek.
3) İlan verisi eksikse bunu açıkça söyle; uydurma bilgi yazma.
4) Kesin hüküm ("kesin alın", "sakın", "dolandırıcı") gibi ifadeler kullanma; risk/şüphe dili kullan.
5) Çıktıyı SADECE JSON döndür.

ÇIKTI ŞABLONU (SADECE BU ALANLAR, ekstra alan ekleme):
{
  "ai_voice": {
    "opening": "2-4 cümlelik insan gibi giriş; araç + profil + risk/maliyet özetini bağlar.",
    "personalized_takeaways": ["4-6 madde; araç ve profile göre değişmeli, tekrar etmemeli"],
    "critical_3_questions": ["Bu araca özel en kritik 3 soru (kısa)"],
    "should_you_go_view": "Eğer kullanıcı soru sorduysa (user_question) bunu yanıtla; yoksa null.",
    "tone_hint": "kısa" 
  }
}

Notlar:
- 'tone_hint' her zaman "kısa" olsun.
- 'should_you_go_view' alanı: user_question boşsa null dön.
"""

SYSTEM_PROMPT_PREMIUM_FULL = """
Sen 'Oto Analiz' uygulaması için **Premium (Detaylı) analiz** üreten ana yapay zekâsın.

AMAÇ:
- Kullanıcı “para boşa gitti” demesin: ilanı gerçekten okuduğunu, kendi aklınla yorumladığını ve kullanıcı profiline göre önceliklendirdiğini hissetsin.
- Gerektiği kadar detay ver: kritik konuları biraz aç, basit konuları kısa geç. Gereksiz tekrar/uzatma yok.

UZUNLUK & ÖNCELİK KURALLARI (ÇOK ÖNEMLİ):
- Metin **gereksiz uzamayacak**. Kullanıcının en çok önemseyeceği 3-5 konuyu seç ve onları biraz daha detaylandır; diğer bölümler kısa/öz.
- "Kesin hüküm" verme: **görünüyor/olası/ekspertizle doğrulanmalı** dili kullan.
- "Alınmaz/tehlikeli/uzak dur/sakın" gibi sert ifadeler YASAK. Bunun yerine: **"Dikkatli değerlendirilmelidir"** veya **"Kontrol şartıyla alınabilir"** kullan.
- **Mini Skorlar bölümü her zaman en üstte olacak ve ASLA atlanmayacak.**

BÖLÜM BAŞINA UZUNLUK SINIRI:
- TL;DR: en fazla **4 cümle**
- Mini Skorlar: her skorun yanında **(6-10 kelimelik gerekçe)** zorunlu
- Güçlü/Zayıf: en fazla **5 madde + 5 madde**
- Araç & İlan Yorumu: en fazla **6 cümle**
- İlan Açıklaması Analizi: en fazla **5 cümle**
- Kronik Sorunlar: en fazla **6 madde** (veya 5-6 kısa cümle)
- Risk Profili: en fazla **3-4 madde**, her madde 1 cümle
- Masraf & Bütçe: 1 satır **Yıllık Toplam**, 2 satır **(Yakıt / Bakım-Onarım)** + 1 kısa yorum
- Şehir & Kullanım: en fazla **5 cümle**
- Parça/Servis: en fazla **3 cümle**
- İkinci El Satış: en fazla **3 cümle**
- Satıcıya Sorular: en fazla **6 soru**
- Kontrol/Ekspertiz: en fazla **8 madde**
- Pazarlık: en fazla **3 cümle**
- Belirsizlikler: en fazla **3 madde**
- Final Karar: 1 satır etiket + en fazla **4 cümle**

FORMAT:
- Başlıkları aynen koru.
- Mini Skorlar'dan sonra 1 satır: **"🎯 En Kritik 3 Nokta:"** ve 3 kısa madde yaz.

TON:
- Samimi ama ciddi. Abartı pazarlama dili yok. Net ve anlaşılır Türkçe.

KURALLAR (çok kritik):
- Uydurma yapma. Girdide olmayan bilgiyi “kesin” gibi söyleme.
- Eksik bilgi varsa bunu açıkça belirt ve “Satıcıya sor” / “Ekspertizde baktır” olarak öner.
- “Kesin al / sakın alma / dolandırıcı” gibi kesin ve suçlayıcı hükümler verme. Koşullu, net yön ver.

GİRDİ:
Tek bir JSON alırsın:
- vehicle, profile, ad_description, context
- enriched: (maliyet/risk/piyasa/uygunluk gibi yapılandırılmış veriler)
- fixed_scores ve fixed_preview

ÇIKTI (SADECE GEÇERLİ JSON):
Aşağıdaki şemaya uy:
{
  "scores": <fixed_scores aynen kopyalanmalı>,
  "summary": {
    "short_comment": string,
    "pros": [string, ...],
    "cons": [string, ...],
    "estimated_risk_level": string
  },
  "preview": <fixed_preview aynen kopyalanmalı>,
  "final_snapshot": {
    "score_because": string,
    "critical_points": [string, string, string]
  },
  "result": string,
  "cards": [
    {"title": string, "content": string},
    ...
  ]
}

ZORUNLU:
- "scores" ve "preview" alanlarını fixed_* alanlarından AYNEN kopyala. Skor/etiket uydurma.
- "result" metninin en başında "📊 Mini Skorlar" bölümü ver:
  * Mekanik Güven, Ekonomi, Satılabilirlik, Kullanıcı Uyumu (X/10) + Risk Seviyesi (Düşük/Orta/Yüksek).
  * Her skor satırının sonunda parantez içinde 6–10 kelimelik kısa gerekçe yaz (ör: "(95 bin km + bakım belirsizliği düşürüyor)").
- Her ana bölüm/kartta en az 1 cümle mutlaka bu araca özel olmalı (km/şehir/ilan detayı/modifiye/kullanıcı profili). Genel konuşma yapma.
- Her card içeriği: madde madde olabilir ama her maddede en az 1 cümle yorum/“neden” olsun (kuru liste yok).
- Uzunluk: Her card genelde 3–8 cümle; sadece “Risk/Kırmızı Bayraklar” veya “Masraf” gerektiğinde biraz daha detaylı olabilir.
- Aynı cümleyi farklı şekilde tekrarlama. Gereksiz teknik detay boğma.

KARTLAR (başlıklar zorunlu; içerik araca/profile göre şekillenir):
1) "🔍 Hızlı Özet (TL;DR)"
   - 2–3 cümle: risk + masraf + gitmeye değer mi (net ama koşullu).

2) "🚗 Araç & İlan Yorumu"
   - Yaş/km/şanzıman/yakıt/ilan sinyalleri + genel ilk izlenim.

3) "📝 İlan Açıklaması Analizi"
   - ad_description içindeki boya/değişen/hasar/küçük notları yorumla.
   - Örn “arka tampon değişti” -> olası senaryo + hangi kontroller istenir (bagaj havuzu/panel/şasi ucu vb.) gibi.

4) "⚠️ Kronik Sorunlar"
   - Bu araç/segment için makul kronikler (varsa) ve BU ilanda riskle ilişkisi.
   - Emin değilsen “model/versiyon net değil → genel kronik ihtimaller” de ve belirsizlik yaz.

5) "🚨 Risk Profili & Kırmızı Bayraklar"
   - En kritik 3 risk. Her biri kısa + neden + nasıl doğrulanır.
   - Gerekirse “gitmeden önce” uyarısı ekle (kapora vs.).

6) "💸 Masraf & Bütçe Uyumu"
   - Bütçe düşük/orta veya masraf hassasiyeti varsa daha detaylı: bakım + olası sürpriz kalemler.
   - Bütçe yüksek ve hassasiyet yoksa özetle.
   - Rakam uydurma: enriched/fixed veriye dayan; yoksa aralık ver ve belirsizliği söyle.

7) "🏙️ Şehir & Kullanım Uyumu"
   - Şehir (örn İstanbul) + kullanım tipi + yıllık km + vites tercihi.
   - Trafik/konfor/otomatik-manuel yorumu (İstanbul’da otomatik avantajı gibi) yap.

8) "🔧 Parça Bulunabilirliği & Servis"
   - Türkiye’de parça/özel servis yaygınlığı ve yetkili servis maliyet trendi (genel, abartısız).
   - Belirsizse “marka/versiyon”a göre temkinli yaz.

9) "📈 İkinci El Satış Kolaylığı"
   - Piyasa talebi: hızlı satılır mı bekletir mi? Neye bağlı?
   - “Fiyat doğruysa/hasar kaydı yoksa/otomatikse” gibi koşullar ekle.

10) "❓ Satıcıya Sorulması Gereken Sorular"
   - 4–6 kritik soru. Kısa, net, araç özel.

11) "📋 Kontrol / Ekspertiz Planı"
   - 6–10 madde. Kısa maddeler: özellikle neye baktırılacak.

12) "💰 Pazarlık Önerisi"
   - Kusur/riske göre pazarlık stratejisi: “şu gerekçeyle şu aralıkta” gibi (rakam yoksa oran/çokluk belirterek).

13) "❔ Belirsizlikler"
   - İlanda eksik kalan en önemli bilgiler ve bunları nasıl netleştireceği.

14) "✅ Final Karar"
   - 1 cümle net karar (gitmeye değer/değmez veya şu şartla mantıklı),
   - 1–2 cümle kısa gerekçe.

EK: Kullanıcı sorusu varsa (context.user_question):
- Ayrı bir card ekle: "💬 Soruna cevap" (2–5 cümle, çok net ve pratik).

OUTPUT KALİTESİ:
- Her card araç/ilan/profil verisine referans versin (en az 1 detayla).
- Kullanıcıyı yormadan “akıllı uzman” gibi konuş.
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


# =========================================================
# QUICK (NORMAL) ANALYZE (deterministic, no LLM)  ✅ yeni
# =========================================================
def _to_100_from_1_5(x: int) -> int:
    try:
        x = int(x)
    except:
        x = 3
    x = _clamp(x, 1, 5)
    return int(round((x - 1) * 25))  # 1->0, 5->100


def quick_analyze_impl(req: AnalyzeRequest) -> Dict[str, Any]:
    """
    'Hızlı Analiz' = Normal analiz.
    - Tek bir genel skor (overall) odaklıdır.
    - Maliyet/Endeks/Profil uyumu kısa ve özet gelir.
    - Puanlamanın ne üzerinden yapıldığını net yazar (kafa karışmasın).
    """
    enriched = build_enriched_context(req)

    v = req.vehicle
    title = f"{v.year or ''} {v.make} {v.model}".strip() or "Hızlı Analiz"

    # temel veriler
    costs = enriched.get("costs", {}) or {}
    risk = enriched.get("risk", {}) or {}
    market = enriched.get("market", {}) or {}
    prof = enriched.get("profile", {}) or {}
    idx = (market.get("indices") or {})

    base_risk = (risk.get("baseline_risk_level") or "orta").strip().lower()
    overall = _score_from_risk(base_risk)

    # belirsizlik -> genel skoru biraz aşağı çekebilir
    uncertainty = build_uncertainty(enriched)
    if (uncertainty or {}).get("level") == "yüksek":
        overall = _clamp(overall - 4, 0, 100)
    elif (uncertainty or {}).get("level") == "orta":
        overall = _clamp(overall - 2, 0, 100)

    # premium / çok karmaşık modeller için tavanı biraz kıs
    seg_code = (enriched.get("segment", {}) or {}).get("code", "C_SEDAN")
    blob = _norm(f"{v.make} {v.model} {(req.ad_description or '')}")
    perf_keys = ["rs", "amg", "m5", "m3", "m4", "m8", "c63", "e63", "911", "gtr", "gt-r", "supra", "sti", "type r", "type-r", "cupra"]
    if seg_code in ("PREMIUM_D", "E_SEGMENT") or any(k in blob for k in perf_keys):
        overall = min(overall, 88)

    # fiyat etiketi (çok kaba)
    listed = costs.get("listed_price_try")
    price_tag = None
    if isinstance(listed, int):
        if listed < 500_000:
            price_tag = "Uygun"
        elif listed < 1_200_000:
            price_tag = "Normal"
        else:
            price_tag = "Yüksek"

    # genel parça-servis puanı
    parts_av = int(idx.get("parts_availability_score_1_5", 3))
    parts_cost = int(idx.get("parts_cost_index_1_5", 3))
    service_net = int(idx.get("service_network_index_1_5", 3))

    availability_100 = _to_100_from_1_5(parts_av)
    service_100 = _to_100_from_1_5(service_net)
    # maliyette 1 iyi, 5 kötü -> tersle
    cost_100 = 100 - _to_100_from_1_5(parts_cost)

    parts_service_100 = int(round((availability_100 + service_100 + cost_100) / 3))

    # kişiye uygunluk (profil)
    fit_100 = _fit_score(req, enriched)

    # yıllık toplam (özet)
    maint_min = int(costs.get("maintenance_yearly_try_min") or 0)
    maint_max = int(costs.get("maintenance_yearly_try_max") or 0)
    fuel_min = int(costs.get("yearly_fuel_tr_min") or 0)
    fuel_max = int(costs.get("yearly_fuel_tr_max") or 0)

    total_min = maint_min + fuel_min if maint_min and fuel_min else None
    total_max = maint_max + fuel_max if maint_max and fuel_max else None
    total_mid = None
    if total_min is not None and total_max is not None:
        total_mid = int((total_min + total_max) / 2)

    # kısa artı/eksi
    pros: List[str] = []
    cons: List[str] = []

    if parts_service_100 >= 70:
        pros.append("Parça/usta erişimi ve servis ağı tarafı genel olarak rahat.")
    else:
        cons.append("Parça/servis maliyet ve erişim tarafı daha dikkatli plan gerektirebilir (özellikle premium sınıfta).")

    if fit_100 >= 78:
        pros.append("Kullanım profiline göre uyumlu bir tablo.")
    elif fit_100 <= 60:
        cons.append("Kullanım profiline göre masraf/uyum riski daha yüksek olabilir (yakıt/şehir içi/segment etkisi).")

    if (uncertainty or {}).get("level") == "yüksek":
        cons.append("Belirsizlik yüksek: tramer + servis kaydı + ekspertiz netleşmeden bandlar geniş kalır.")

    # ✅ BUGFIX: segment_label tanımlı olsun
    segment_label = (enriched.get("segment", {}) or {}).get("name") or (enriched.get("segment", {}) or {}).get("code") or ""
    target_line = infer_target_user_line(segment_label, req.vehicle.make, req.vehicle.model)

    # -------------------------
    # Olası sorunlar (kronik/dikkat) – LLM yok, sadece kural bazlı kısa ipuçları
    # -------------------------
    possible_issues: List[str] = []
    fuel_s = (getattr(v, "fuel", None) or getattr(req.profile, "fuel_preference", None) or "").strip().lower()
    usage_s = (getattr(req.profile, "usage", None) or "mixed").strip().lower()
    trn_s = _infer_transmission(req) or ""
    km = int(getattr(v, "mileage_km", 0) or 0)

    # metinden ipucu (DSG/CVT/LPG vb.)
    blob = f"{req.ad_description or ''} {json.dumps(req.context or {}, ensure_ascii=False)} {v.make} {v.model}"
    tnorm = _norm(blob)

    if fuel_s == "diesel" and usage_s in ("city", "şehir", "sehir"):
        possible_issues.append("DPF/EGR: şehir içi kısa mesafe ağırlıktaysa kurum/arıza riski artar (rejenerasyon geçmişi önemli).")
    if fuel_s == "lpg":
        possible_issues.append("LPG: montaj/ayar kalitesi kritik; subap/ateşleme ve regülatör-enjektör bakımı kontrol edilmeli.")

    if trn_s == "automatic":
        if any(x in tnorm for x in ["dsg", "s tronic", "s-tronic", "dq200", "dq250", "mechatronic", "mekatronik"]):
            possible_issues.append("DSG/S-Tronic: mekatronik + kavrama durumu ve yağ/bakım geçmişi kritik (test sürüşünde vuruntu/kararsız geçiş bak).")
        elif "cvt" in tnorm:
            possible_issues.append("CVT: şanzıman yağı/bakımı ve kayış/konvertör sağlığı önemli (yük altında bağırma/kaçırma kontrolü).")
        else:
            possible_issues.append("Otomatik şanzıman: yağ/bakım geçmişi önemli; test sürüşünde vuruntu/gecikme ve kaçak kontrolü yapılmalı.")
    elif trn_s == "manual":
        if km >= 120000:
            possible_issues.append("Manuel şanzıman: km yükseldikçe debriyaj/volan yıpranması öne çıkar (kavrama, titreme, kaçırma kontrolü).")

    if km >= 150000:
        possible_issues.append("Yüksek km: alt takım, turbo/soğutma ve enjektör-yakıt sistemi gibi kalemlerde masraf olasılığı yükselir.")
    if getattr(v, "year", None) and int(v.year or 0) <= 2012:
        possible_issues.append("Yaşlı araç: sensör/elektronik ve trim detaylarında küçük arızalar daha sık görülebilir.")

    # çok uzamasın
    possible_issues = [x for x in possible_issues if x][:4]

    # skor metodolojisi (kısa)
    how_scored = [
        "- **Segment & karmaşıklık:** Premium/performans sınıfında parça-işçilik ve elektronik yoğunluğu skor tavanını düşürür.",
        "- **Yaş & km:** Yaş/km arttıkça büyük bakım riski artar; skor aşağı iner.",
        "- **Yakıt & kullanım:** Şehir içi + dizel + düşük km gibi kombinasyonlar risk artırabilir.",
        "- **Parça/servis & 2.el endeksleri:** Bulunurluk, maliyet ve servis ağı genel puanı etkiler.",
        "- **Belirsizlik:** İlan bilgisi azsa (tramer/bakım yoksa) skor daha temkinli verilir.",
    ]

    short_comment = (f"Genel skor **{overall}/100**. Parça/servis geneli **{parts_service_100}/100**, "
                    f"kişiye uygunluk **{fit_100}/100** (tahmini).")

    result_lines: List[str] = []
    result_lines.append(f"## {title}")
    result_lines.append(short_comment)
    result_lines.append("")
    result_lines.append("## Skorlar")
    result_lines.append(f"- Genel skor: **{overall}/100**")
    result_lines.append(f"- Parça/servis geneli: **{parts_service_100}/100**")
    result_lines.append(f"- Kişiye uygunluk: **{fit_100}/100**")
    result_lines.append("")
    if total_mid is not None and total_min is not None and total_max is not None:
        result_lines.append("## Yıllık maliyet (bakım + yakıt)")
        result_lines.append(f"- Tahmini band: **{_fmt_try(total_min)} – {_fmt_try(total_max)} TL** (orta: ~{_fmt_try(total_mid)} TL)")
        result_lines.append("")
    result_lines.append("### Bu skor neye göre verildi?")
    result_lines.extend(how_scored)
    result_lines.append("")
    result_lines.append("### Kime daha uygun?")
    result_lines.append(f"- {target_line}")
    result_lines.append("")
    # -------------------------
    # Olası sorunlar bölümü (kronik/dikkat)
    # -------------------------
    result_lines.append("### Olası sorunlar (kronik/dikkat)")
    if possible_issues:
        for it in possible_issues:
            result_lines.append(f"- {it}")
    else:
        result_lines.append("- Belirgin bir kronik sinyali yok; asıl belirleyici **bakım geçmişi + ekspertiz** sonucu olur.")
    result_lines.append("")
    result_lines.append("")
    result_lines.append("### Hızlı kontrol (en kritik 3)")
    result_lines.append("- Tramer/hasar + şasi/podye kontrolü")
    result_lines.append("- Servis/bakım geçmişi (fatura/kayıt)")
    result_lines.append("- Test sürüşü + OBD taraması")
    result = "\n".join(result_lines)

    # Normal endpoint şemasını koru (UI kırılmasın)
    return {
        "scores": {"overall_100": overall, "mechanical_100": overall, "body_100": overall, "economy_100": parts_service_100},
        "summary": {
            "short_comment": short_comment,
            "pros": pros[:3],
            "cons": cons[:3],
            "estimated_risk_level": base_risk if base_risk in ("düşük","orta","orta-yüksek","yüksek") else "orta",
        },
        "preview": {
            "title": title,
            "price_tag": price_tag,
            "spoiler": "Genel skor, yıllık toplam band, parça/servis genel puanı ve profil uyumu",
            "bullets": ["Genel skor + kısa gerekçe", "Yıllık toplam (bakım+yakıt) bandı", "Parça/servis geneli", "Kime uygun?"],
        },
        "result": result,
    }


def call_llm_or_fallback(model_name: str, system_prompt: str, user_content: str, fallback_fn, req_obj):
    out = call_llm_json(model_name, system_prompt, user_content)
    if isinstance(out, dict):
        return out
    return fallback_fn(req_obj)


def _extract_user_question(req: AnalyzeRequest) -> Optional[str]:
    # Prefer context.user_question, then profile.user_question, then context.question
    try:
        q = None
        if isinstance(getattr(req, "context", None), dict):
            q = req.context.get("user_question") or req.context.get("question")
        if not q and getattr(req, "profile", None) is not None:
            q = getattr(req.profile, "user_question", None) or getattr(req.profile, "question", None)
        q = (q or "").strip()
        return q if q else None
    except Exception:
        return None

def premium_enrich_with_llm(base_report: Dict[str, Any], req: AnalyzeRequest) -> Optional[Dict[str, Any]]:
    """
    Premium raporu bozmaz; sadece ek bir 'ai_voice' alanı döner.
    Flutter tarafı isterse gösterir, istemezse yok sayar.
    """
    try:
        v = req.vehicle
        p = req.profile
        q = _extract_user_question(req)

        # Kart içerikleri çok uzun; LLM'e token yedirmemek için kırpıyoruz.
        cards = base_report.get("cards") or []
        cards_compact = []
        if isinstance(cards, list):
            for c in cards[:8]:
                if isinstance(c, dict):
                    cards_compact.append({
                        "title": c.get("title"),
                        "content": (str(c.get("content") or "")[:900])
                    })

        user_content = {
            "vehicle": v.dict(),
            "profile": p.dict(),
            "user_question": q,
            "base": {
                "scores": base_report.get("scores"),
                "summary": base_report.get("summary"),
                "costs": base_report.get("costs"),
                "segment": base_report.get("segment"),
                "uncertainty": base_report.get("uncertainty"),
                "cards_compact": cards_compact,
            },
            "rules": "Kesin hüküm verme; eksik bilgi varsa söyle; uydurma yapma; araç+profil özel yaz."
        }

        out = call_llm_json(
            OPENAI_MODEL_PREMIUM,
            SYSTEM_PROMPT_PREMIUM_ENRICH,
            json.dumps(user_content, ensure_ascii=False),
        )
        if isinstance(out, dict) and isinstance(out.get("ai_voice"), dict):
            return {"ai_voice": out.get("ai_voice")}
        return None
    except Exception as e:
        print("premium_enrich_with_llm error:", e)
        return None


# =========================================================
# HEALTH
# =========================================================
@app.get("/")
async def root() -> Dict[str, Any]:
    return {"ok": True, "message": "Oto Analiz backend çalışıyor."}


# =========================================================
# ENDPOINTS
# =========================================================
# =========================================================
# VEHICLE COVER CACHE (no user photo) + WATERMARK
# =========================================================

class CoverRequest(BaseModel):
    model_config = ConfigDict(extra='ignore')
    brand: str
    model: str
    body: Optional[str] = None
    generation: Optional[str] = None
    year: Optional[int] = None
    color: Optional[str] = None
    cache_key: Optional[str] = None
    force_regenerate: bool = False

class CoverResponse(BaseModel):
    coverKey: str
    imageUrl: str
    cached: bool


def _year_bucket(year: Optional[int]) -> str:
    if not year:
        return "unknown"
    base = (year // 5) * 5
    return f"{base}-{base+4}"

def _norm_color(color: Optional[str]) -> str:
    c = (color or "").strip().lower()
    mp = {
        "beyaz": "white",
        "siyah": "black",
        "gri": "gray",
        "mavi": "blue",
        "lacivert": "blue",
        "kırmızı": "red",
        "kirmizi": "red",
        "yeşil": "green",
        "yesil": "green",
        "gümüş": "silver",
        "gumus": "silver",
    }
    return mp.get(c, c or "any")

def _cover_key(
    brand: str,
    model: str,
    body: Optional[str] = None,
    year: Optional[int] = None,
    color: Optional[str] = None,
    generation: Optional[str] = None,
) -> str:
    """Cache key rule (critical):

    New cover is generated ONLY if one of these changes:
    - brand, model
    - body (sedan/suv/hatch etc.)
    - generation/kasa (if provided)
    - year
    - color
    """
    # lightweight normalization for cache stability
    def _norm_token(x: Optional[str]) -> str:
        x = (x or "").strip().lower()
        x = x.replace("ı", "i").replace("ş", "s").replace("ğ", "g").replace("ü", "u").replace("ö", "o").replace("ç", "c")
        x = re.sub(r"\s+", "_", x)
        x = re.sub(r"[^a-z0-9_\-|]", "", x)
        return x

    # brand aliases (typo tolerance)
    b = _norm_token(brand)
    brand_alias = {
        "mercdes": "mercedes",
        "mercedez": "mercedes",
        "mercedesbenz": "mercedes",
        "mercedes-benz": "mercedes",
        "benz": "mercedes",
        "mb": "mercedes",
        "vw": "volkswagen",
    }
    b = brand_alias.get(b, b)

    m = _norm_token(model)

    parts = [b, m]

    if body:
        parts.append(_norm_token(body))

    if generation:
        parts.append(_norm_token(generation))

    # exact year (not bucket) — user rule
    if isinstance(year, int) and 1900 <= year <= 2100:
        parts.append(str(year))
    else:
        parts.append("unknown_year")

    parts.append(_norm_color(color))

    key = "|".join([p for p in parts if p])
    key = re.sub(r"\|{2,}", "|", key).strip("|")
    return key or "unknown"




def _normalize_cover_key(key: str) -> str:
    k = (key or '').strip().lower()
    k = re.sub(r"\s+", "-", k)
    k = re.sub(r"[^a-z0-9\-|_]", "", k)
    return k or "unknown"

def _init_firebase_admin() -> None:
    if firebase_admin is None or fb_credentials is None or fb_firestore is None or fb_storage is None:
        raise HTTPException(status_code=500, detail="firebase_admin_not_installed")

    # Already initialized?
    try:
        if getattr(firebase_admin, "_apps", None) and len(firebase_admin._apps) > 0:  # type: ignore
            return
    except Exception:
        pass

    bucket = (os.getenv("FIREBASE_STORAGE_BUCKET") or "").strip()
    if not bucket:
        raise HTTPException(status_code=500, detail="missing_FIREBASE_STORAGE_BUCKET")

    sa_b64 = (os.getenv("FIREBASE_SERVICE_ACCOUNT_B64") or "").strip()
    sa_json = (os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON") or "").strip()

    # allow GOOGLE_APPLICATION_CREDENTIALS to point to a JSON file path (fallback)
    if not sa_json:
        sa_json = (os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or "").strip()

    if not sa_b64 and not sa_json:
        raise HTTPException(
            status_code=500,
            detail="missing_FIREBASE_SERVICE_ACCOUNT_B64_and_FIREBASE_SERVICE_ACCOUNT_JSON",
        )

    try:
        if sa_b64:
            decoded = base64.b64decode(sa_b64).decode("utf-8")
            cred = fb_credentials.Certificate(json.loads(decoded))
        else:
            if sa_json.startswith("{"):
                cred = fb_credentials.Certificate(json.loads(sa_json))
            else:
                cred = fb_credentials.Certificate(sa_json)  # path
        firebase_admin.initialize_app(cred, {"storageBucket": bucket})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"firebase_init_failed: {e}")


def _firebase_cover_doc(cover_key: str):
    _init_firebase_admin()
    db = fb_firestore.client()
    return db.collection("discover_covers_cache").document(cover_key)


def _firebase_upload_cover_bytes(
    cover_key: str,
    image_bytes: bytes,
    content_type: str = "image/webp",
    object_path: Optional[str] = None,
) -> str:
    """Upload bytes to Firebase Storage and return a token-based download URL.

    - Default path stays backward compatible: discover_covers/generated/{cover_key}.webp
    - For multi-variant covers you can pass object_path, e.g.
      discover_covers/generated/{cover_key}/bg_07.webp
    """
    _init_firebase_admin()
    bucket_name = os.getenv("FIREBASE_STORAGE_BUCKET", "").strip()
    bucket = fb_storage.bucket()  # default bucket from initialize_app

    if not object_path:
        object_path = f"discover_covers/generated/{cover_key}.webp"

    token = uuid.uuid4().hex
    blob = bucket.blob(object_path)
    blob.metadata = {"firebaseStorageDownloadTokens": token}
    blob.upload_from_string(image_bytes, content_type=content_type)
    blob.patch()

    # Firebase download URL (token-based)
    url = f"https://firebasestorage.googleapis.com/v0/b/{bucket_name}/o/{urlquote(object_path, safe='')}?alt=media&token={token}"
    return url



def _fix_firebase_download_url(url: str) -> str:
    """Fix Firebase Storage download URLs where the object name is not fully URL-encoded (slashes not encoded).
    Some URLs may contain `/o/folder/file.webp` instead of `/o/folder%2Ffile.webp`, which causes HTTP 400.
    We safely fix this by encoding only the slashes in the object part."""
    try:
        if not url or "firebasestorage.googleapis.com" not in url or "/o/" not in url:
            return url
        pre, rest = url.split("/o/", 1)
        if "?" not in rest:
            return url
        obj, qs = rest.split("?", 1)
        # If already encoded, keep as-is
        if "%2F" in obj or "%2f" in obj:
            return url
        if "/" in obj:
            obj_fixed = obj.replace("/", "%2F")
            return f"{pre}/o/{obj_fixed}?{qs}"
        return url
    except Exception:
        return url


def _brand_design_cues(brand: str) -> str:
    b = (brand or "").strip().lower()
    cues = {
        "bmw": "Signature twin kidney grille (no logo), sporty proportions, sharp LED headlights, Hofmeister kink on the rear window line.",
        "mercedes": "Elegant premium sedan design, wide grille shape with a central mount area (no logo), smooth sculpted hood, refined LED headlights.",
        "mercedes-benz": "Elegant premium sedan design, wide grille shape with a central mount area (no logo), smooth sculpted hood, refined LED headlights.",
        "audi": "Large single-frame hexagonal grille (no logo), crisp shoulder line, modern LED headlights, clean German design language.",
        "volkswagen": "Simple horizontal grille (no logo), understated clean lines, practical compact sedan/hatch styling.",
        "vw": "Simple horizontal grille (no logo), understated clean lines, practical compact sedan/hatch styling.",
        "honda": "Sleek Japanese compact design, slim headlights, clean aerodynamic body lines.",
        "toyota": "Modern Japanese design, sharp headlights, balanced proportions, clean body surfacing.",
        "ford": "Modern mass-market design, trapezoid grille shape (no logo), confident stance, practical proportions.",
        "opel": "European compact design, clean surfaces, modern headlights, restrained styling.",
        "renault": "European design language, modern headlights, rounded-yet-sharp surfaces, compact proportions.",
        "peugeot": "Modern French design, sharp DRL signature, confident grille shape (no logo).",
        "hyundai": "Bold modern design, sharp LED DRLs, geometric grille shape (no logo), clean surfacing.",
        "kia": "Modern design, tiger-nose grille shape (no logo), sharp LED headlights, sporty proportions.",
        "skoda": "Czech design language, crystalline headlight style, practical proportions, clean lines.",
        "seat": "Sporty Spanish design, sharp angles, compact proportions, modern headlights.",
        "fiat": "Compact city-car design, friendly rounded surfaces, simple detailing.",
        "citroen": "Distinctive French design, split headlight signature, playful shapes.",
    }
    # fallbacks: match contains
    for k,v in cues.items():
        if b == k or b.startswith(k) or k in b:
            return v
    return "Make the design clearly match the brand and model described."

def _generate_vehicle_image_bytes(
    brand: str,
    model: str,
    body: Optional[str] = None,
    generation: Optional[str] = None,
    year: Optional[int] = None,
    color: Optional[str] = None,
) -> bytes:
    """Generate a single-car exterior cover image using OpenAI Images.

    We intentionally avoid stock-photo search (Pexels etc.) because it can return
    wrong brands/models. Instead we generate a consistent, controllable cover.

    Output: raw image bytes (PNG) from the Images API. We later watermark + convert to WEBP.
    """
    if client is None:
        raise HTTPException(status_code=500, detail="openai_client_not_configured")

    b = (brand or "").strip()
    m = (model or "").strip()
    if not b or not m:
        raise HTTPException(status_code=400, detail="missing_brand_or_model")

    # Normalize very common typos/aliases (for both prompt & cache stability)
    brand_alias = {
        "mercedesbenz": "Mercedes-Benz",
        "mercedes-benz": "Mercedes-Benz",
        "mercdes": "Mercedes-Benz",
        "mercedez": "Mercedes-Benz",
        "mercedes": "Mercedes-Benz",
        "benz": "Mercedes-Benz",
        "mb": "Mercedes-Benz",
        "bmw": "BMW",
        "volkswagen": "Volkswagen",
        "vw": "Volkswagen",
        "audi": "Audi",
        "toyota": "Toyota",
        "honda": "Honda",
        "hyundai": "Hyundai",
        "kia": "Kia",
        "ford": "Ford",
        "renault": "Renault",
        "fiat": "Fiat",
        "peugeot": "Peugeot",
        "citroen": "Citroën",
        "opel": "Opel",
        "skoda": "Škoda",
        "seat": "SEAT",
        "dacia": "Dacia",
        "nissan": "Nissan",
        "mazda": "Mazda",
        "volvo": "Volvo",
        "porsche": "Porsche",
    }

    b_key = re.sub(r"[^a-z0-9]+", "", b.lower())
    b_norm = brand_alias.get(b_key, b.strip().title())

    # Model cleanup
    m_norm = re.sub(r"\s+", " ", m.strip())

    # Color TR->EN (prompt only; cache uses _norm_color elsewhere)
    def _color_en(x: Optional[str]) -> str:
        if not x:
            return ""
        mp = {
            "siyah": "black",
            "beyaz": "white",
            "gri": "gray",
            "füme": "gray",
            "fume": "gray",
            "kırmızı": "red",
            "kirmizi": "red",
            "mavi": "blue",
            "lacivert": "navy blue",
            "yeşil": "green",
            "yesil": "green",
            "sarı": "yellow",
            "sari": "yellow",
            "turuncu": "orange",
            "mor": "purple",
            "bej": "beige",
            "kahverengi": "brown",
            "gümüş": "silver",
            "gumus": "silver",
        }
        xx = (x or "").strip().lower()
        xx = xx.replace("ı", "i").replace("ş", "s").replace("ğ", "g").replace("ü", "u").replace("ö", "o").replace("ç", "c")
        return mp.get(xx, xx)

    c_en = _color_en(color)

    # Prompt: photoreal, single car, clean background, front 3/4
    # Also explicitly forbid extra objects/people/text/logos.
    y = str(int(year)) if isinstance(year, int) and 1950 <= int(year) <= 2035 else ""
    body_s = (body or "").strip()
    gen_s = (generation or "").strip()

    # Brand-specific design cues help reduce "wrong brand" look.
    cues = _brand_design_cues(b_norm)

    parts = []
    if c_en:
        parts.append(c_en)
    if y:
        parts.append(y)
    parts.append(b_norm)
    parts.append(m_norm)
    if gen_s:
        parts.append(gen_s)
    if body_s:
        parts.append(body_s)

    subject = " ".join([p for p in parts if p]).strip()

    prompt = (
        "High-end automotive studio photograph. "
        "Medium-dark charcoal/graphite gradient background. "
        "The car is parked on a clean asphalt-like ground surface with subtle texture; realistic tire contact shadow. "
        f"Photorealistic exterior of a single {subject}. "
        "Centered composition, sharp focus, realistic proportions, natural reflections. "
        "Front three-quarter view (or straight front if needed), 50mm lens look. "
        f"Must match the real-world {b_norm} {m_norm} front fascia for the given year/generation; "
        "headlights and grille shape must be consistent with that exact model/year. "
        "Do NOT generate a generic car. "
        "No people, no interior, no text, no watermark, no showroom/dealership, no extra objects. "
        "Avoid visible brand logos/badges, but keep the correct silhouette and design cues. "
        f"Only one car in the frame. {cues}"
    )

    image_model = (os.getenv("OPENAI_IMAGE_MODEL") or "gpt-image-1").strip()

    # Cost rule: we NEVER use medium/high in production (only low).
    image_quality = "low"

    # Size is fixed per product decision
    size = "1024x1024"

    # OpenAI Python SDK returns base64 for images
    res = client.images.generate(
        model=image_model,
        prompt=prompt,
        size=size,
        quality=image_quality,
    )

    # Support both shapes (b64_json or raw bytes-like)
    b64 = None
    try:
        if hasattr(res, "data") and res.data and getattr(res.data[0], "b64_json", None):
            b64 = res.data[0].b64_json
        elif isinstance(res, dict):
            # Very defensive for older/alt shapes
            b64 = (((res.get("data") or [{}])[0]).get("b64_json"))
    except Exception:
        b64 = None

    if not b64:
        raise RuntimeError("missing_b64_json_from_images_api")

    return base64.b64decode(b64)

def _apply_watermark(
    image_bytes: bytes,
    watermark_text: str = "Oto Analiz",
    bg_id: int = 1,
) -> bytes:
    """Post-process cover image (PIL).

    What we guarantee:
    - Medium/dark background with an asphalt/concrete ground plane (car never "floating")
    - Single sahibinden-style watermark: "Oto Analiz" (once)
    - Small AI badge at top-right

    bg_id (1..24) controls the background variant.

    Always returns WEBP bytes. If PIL is unavailable, returns the original bytes.
    """
    if Image is None or ImageDraw is None or ImageFont is None:
        return image_bytes

    try:
        from PIL import ImageChops, ImageFilter
    except Exception:
        ImageChops = None  # type: ignore
        ImageFilter = None  # type: ignore

    # --- Load
    im_rgba = Image.open(BytesIO(image_bytes)).convert("RGBA")
    W, H = im_rgba.size

    # --- Background variants (24): keep as lightweight parameters (not 24 huge prompts)
    # Each entry: (topRGB, midRGB, bottomRGB, asphaltRGB, accent_mode)
    # accent_mode: None | "bokeh" | "streak" | "garage" | "rain"
    BG = {
        1:  ((46, 50, 56), (28, 31, 36), (18, 20, 24), (22, 24, 28), None),
        2:  ((55, 58, 62), (30, 32, 36), (16, 18, 22), (20, 22, 26), "garage"),
        3:  ((40, 46, 58), (22, 26, 36), (14, 16, 22), (18, 20, 26), "bokeh"),
        4:  ((60, 55, 52), (32, 30, 30), (18, 18, 20), (22, 21, 22), None),
        5:  ((44, 52, 50), (24, 30, 30), (14, 16, 18), (18, 19, 20), "streak"),
        6:  ((54, 44, 48), (30, 24, 28), (18, 14, 16), (20, 18, 18), None),
        7:  ((42, 48, 62), (22, 26, 38), (12, 14, 22), (16, 18, 26), "rain"),
        8:  ((58, 60, 64), (30, 32, 34), (16, 18, 20), (22, 24, 26), None),
        9:  ((48, 44, 40), (28, 24, 22), (16, 14, 14), (20, 18, 18), "garage"),
        10: ((46, 54, 46), (26, 32, 26), (16, 20, 16), (20, 22, 20), None),
        11: ((50, 52, 60), (26, 28, 34), (14, 16, 22), (18, 20, 26), "bokeh"),
        12: ((52, 48, 58), (28, 26, 34), (16, 14, 22), (20, 18, 26), "streak"),
        13: ((62, 58, 50), (34, 32, 28), (18, 18, 18), (22, 21, 20), None),
        14: ((40, 42, 48), (22, 24, 28), (12, 14, 18), (18, 20, 22), "garage"),
        15: ((44, 50, 66), (24, 28, 40), (14, 16, 24), (18, 20, 26), "bokeh"),
        16: ((60, 50, 46), (32, 28, 26), (18, 16, 16), (22, 20, 20), None),
        17: ((50, 56, 54), (26, 32, 30), (14, 18, 18), (20, 22, 22), "rain"),
        18: ((56, 56, 58), (30, 30, 32), (16, 16, 18), (22, 22, 24), None),
        19: ((46, 46, 54), (24, 24, 34), (14, 14, 22), (18, 18, 26), "streak"),
        20: ((52, 44, 44), (28, 24, 24), (16, 14, 14), (20, 18, 18), None),
        21: ((44, 58, 58), (24, 34, 34), (14, 20, 20), (18, 22, 22), "bokeh"),
        22: ((58, 52, 62), (30, 28, 36), (16, 16, 22), (22, 20, 26), None),
        23: ((42, 44, 54), (22, 24, 34), (12, 14, 22), (18, 20, 26), "rain"),
        24: ((54, 54, 54), (30, 30, 30), (16, 16, 16), (22, 22, 22), None),
    }

    if bg_id not in BG:
        try:
            bg_id = int(bg_id)
        except Exception:
            bg_id = 1
    if bg_id not in BG:
        bg_id = 1

    top, mid, bottom, asphalt_rgb, accent = BG[bg_id]

    # --- Helper: corner brightness to decide if background-keying makes sense
    def _avg_corner_brightness(img: "Image.Image") -> float:
        rgb = img.convert("RGB")
        samples = []
        step = max(1, int(min(W, H) * 0.02))
        boxes = [
            (0, 0, step, step),
            (W - step, 0, W, step),
            (0, H - step, step, H),
            (W - step, H - step, W, H),
        ]
        for b in boxes:
            crop = rgb.crop(b)
            px = list(crop.getdata())
            if not px:
                continue
            lum = [(0.2126*r + 0.7152*g + 0.0722*b) for (r, g, b) in px]
            samples.append(sum(lum) / len(lum))
        return sum(samples) / max(1, len(samples))

    # --- Build background (variant gradient + asphalt)
    def _build_background() -> "Image.Image":
        bg = Image.new("RGB", (W, H), bottom)
        draw = ImageDraw.Draw(bg)

        # vertical gradient (top -> mid -> bottom)
        split = 0.66
        for y in range(H):
            t = y / float(max(1, H - 1))
            if t < split:
                k = t / split
                r = int(top[0] + (mid[0] - top[0]) * k)
                g = int(top[1] + (mid[1] - top[1]) * k)
                b = int(top[2] + (mid[2] - top[2]) * k)
            else:
                k = (t - split) / float(max(1e-6, 1.0 - split))
                r = int(mid[0] + (bottom[0] - mid[0]) * k)
                g = int(mid[1] + (bottom[1] - mid[1]) * k)
                b = int(mid[2] + (bottom[2] - mid[2]) * k)
            draw.line([(0, y), (W, y)], fill=(r, g, b))

        asphalt_y = int(H * 0.70)
        asphalt_h = H - asphalt_y
        asphalt = Image.new("RGB", (W, asphalt_h), asphalt_rgb)

        # texture (noise)
        try:
            noise = Image.effect_noise((W, asphalt_h), 64).convert("L")
            if ImageFilter is not None:
                noise = noise.filter(ImageFilter.GaussianBlur(radius=1.2))
            noise = noise.point(lambda p: int(p * 0.18))
            tex = Image.new("RGB", (W, asphalt_h), (0, 0, 0))
            tex.putalpha(noise)
            asphalt_rgba = asphalt.convert("RGBA")
            asphalt_rgba = Image.alpha_composite(asphalt_rgba, tex)
            asphalt = asphalt_rgba.convert("RGB")
        except Exception:
            pass

        # subtle vignette
        try:
            vign = Image.new("L", (W, asphalt_h), 0)
            vdraw = ImageDraw.Draw(vign)
            for i in range(0, max(W, asphalt_h), 8):
                alpha = int(110 * (i / float(max(1, max(W, asphalt_h)))))
                vdraw.rectangle([i, i, W - i, asphalt_h - i], outline=alpha)
            if ImageFilter is not None:
                vign = vign.filter(ImageFilter.GaussianBlur(radius=18))
            vcol = Image.new("RGBA", (W, asphalt_h), (0, 0, 0, 0))
            vcol.putalpha(vign)
            asphalt = Image.alpha_composite(asphalt.convert("RGBA"), vcol).convert("RGB")
        except Exception:
            pass

        bg.paste(asphalt, (0, asphalt_y))

        # lightweight accents for variety (kept subtle so car remains focus)
        try:
            if accent == "bokeh" and ImageFilter is not None:
                dots = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                d = ImageDraw.Draw(dots)
                for i in range(18):
                    x = int((W * (i * 37 % 100)) / 100.0)
                    y = int((H * (i * 53 % 100)) / 140.0)
                    r = int(min(W, H) * (0.03 + (i % 5) * 0.006))
                    a = 18 + (i % 7) * 6
                    d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255, a))
                dots = dots.filter(ImageFilter.GaussianBlur(radius=8))
                bg = Image.alpha_composite(bg.convert("RGBA"), dots).convert("RGB")

            if accent == "streak" and ImageFilter is not None:
                st = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                d = ImageDraw.Draw(st)
                for i in range(6):
                    y = int(H * (0.12 + i * 0.07))
                    d.rectangle([int(W*0.15), y, int(W*0.85), y + 2], fill=(255, 255, 255, 18))
                st = st.filter(ImageFilter.GaussianBlur(radius=6))
                bg = Image.alpha_composite(bg.convert("RGBA"), st).convert("RGB")

            if accent == "garage" and ImageFilter is not None:
                g = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                d = ImageDraw.Draw(g)
                # soft vertical columns
                for i in range(5):
                    x = int(W * (0.12 + i * 0.18))
                    d.rectangle([x, 0, x + int(W*0.05), int(H*0.68)], fill=(0, 0, 0, 22))
                g = g.filter(ImageFilter.GaussianBlur(radius=10))
                bg = Image.alpha_composite(bg.convert("RGBA"), g).convert("RGB")

            if accent == "rain" and ImageFilter is not None:
                rimg = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                d = ImageDraw.Draw(rimg)
                for i in range(70):
                    x = int((i * 97) % W)
                    y0 = int((i * 53) % int(H*0.68))
                    d.line([x, y0, x + 8, y0 + 18], fill=(255, 255, 255, 10), width=1)
                rimg = rimg.filter(ImageFilter.GaussianBlur(radius=1))
                bg = Image.alpha_composite(bg.convert("RGBA"), rimg).convert("RGB")
        except Exception:
            pass

        return bg

    bg = _build_background()

    # --- If background is bright (white studio), key it out and re-place car
    try:
        corner_b = _avg_corner_brightness(im_rgba)
    except Exception:
        corner_b = 0

    composed = bg.convert("RGBA")

    if corner_b >= 220 and ImageChops is not None:
        try:
            rgb = im_rgba.convert("RGB")
            diff = ImageChops.difference(rgb, Image.new("RGB", (W, H), (255, 255, 255))).convert("L")
            mask = diff.point(lambda p: 255 if p > 18 else 0)
            if ImageFilter is not None:
                mask = mask.filter(ImageFilter.GaussianBlur(radius=1.6))

            bbox = mask.getbbox()
            if bbox:
                pad = int(min(W, H) * 0.02)
                x0, y0, x1, y1 = bbox
                x0 = max(0, x0 - pad)
                y0 = max(0, y0 - pad)
                x1 = min(W, x1 + pad)
                y1 = min(H, y1 + pad)

                car_crop = im_rgba.crop((x0, y0, x1, y1))
                mask_crop = mask.crop((x0, y0, x1, y1))
                cw, ch = car_crop.size

                desired_bottom = int(H * 0.93)
                px = (W - cw) // 2
                py = desired_bottom - ch
                py = max(0, min(H - ch, py))

                # shadow (subtle, helps "ground contact")
                if ImageFilter is not None:
                    sh = mask_crop.filter(ImageFilter.GaussianBlur(radius=10))
                else:
                    sh = mask_crop
                sh = sh.point(lambda p: int(p * 0.35))
                shadow = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
                shadow.putalpha(sh)
                composed.alpha_composite(shadow, (px, min(H - ch, py + 10)))

                composed.alpha_composite(car_crop, (px, py), mask_crop)
            else:
                composed.alpha_composite(im_rgba)
        except Exception:
            composed.alpha_composite(im_rgba)
    else:
        composed = im_rgba

    # --- Overlays: watermark + AI badge
    over = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(over)

    def _load_font(size: int):
        try:
            for fp in (
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
            ):
                if os.path.exists(fp):
                    return ImageFont.truetype(fp, size=size)
        except Exception:
            pass
        try:
            return ImageFont.load_default()
        except Exception:
            return None

    # 1) Watermark (single)
    wm_text = (watermark_text or "Oto Analiz").strip() or "Oto Analiz"
    wm_text = wm_text.upper() if len(wm_text) <= 16 else wm_text

    font_size = max(52, int(min(W, H) * 0.11))
    font = _load_font(font_size)
    if font is not None:
        try:
            bbox = draw.textbbox((0, 0), wm_text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
        except Exception:
            tw, th = (font_size * max(6, len(wm_text)//2), font_size)

        cx = (W - tw) // 2
        cy = (H - th) // 2
        draw.text((cx, cy), wm_text, font=font, fill=(255, 255, 255, 30))
        try:
            over = over.rotate(-16, resample=Image.BICUBIC, expand=0)
        except Exception:
            pass

    # 2) AI badge (top-right)
    badge_pad = int(min(W, H) * 0.03)
    badge_w = int(min(W, H) * 0.14)
    badge_h = int(min(W, H) * 0.07)
    bx1 = W - badge_pad
    by0 = badge_pad
    bx0 = bx1 - badge_w
    by1 = by0 + badge_h

    try:
        draw = ImageDraw.Draw(over)
        if hasattr(draw, "rounded_rectangle"):
            draw.rounded_rectangle([bx0, by0, bx1, by1], radius=int(badge_h * 0.28), fill=(255, 255, 255, 210))
        else:
            draw.rectangle([bx0, by0, bx1, by1], fill=(255, 255, 255, 210))
    except Exception:
        pass

    ai_font = _load_font(max(18, int(badge_h * 0.55)))
    if ai_font is not None:
        try:
            t = "AI"
            tb = draw.textbbox((0, 0), t, font=ai_font)
            tw = tb[2] - tb[0]
            th = tb[3] - tb[1]
            tx = bx0 + (badge_w - tw) // 2
            ty = by0 + (badge_h - th) // 2
            draw.text((tx, ty), t, font=ai_font, fill=(0, 0, 0, 235))
        except Exception:
            draw.text((bx0 + 10, by0 + 6), "AI", font=ai_font, fill=(0, 0, 0, 235))

    base = composed.convert("RGBA")
    out_rgba = Image.alpha_composite(base, over)
    out_rgb = out_rgba.convert("RGB")

    buf = BytesIO()
    out_rgb.save(buf, format="WEBP", quality=90)
    return buf.getvalue()


@app.post("/get_or_create_cover")
def get_or_create_cover(req: CoverRequest) -> Dict[str, Any]:
    """Get or create a vehicle cover with multi-variant cache.

    Rules (final):
    - 24 background pool
    - Per vehicle (cover_key) keep max 3 variants
    - If <3: generate a NEW variant with an unused background id
    - If 3 full: serve one of existing 3 (no cost), avoid repeating last served
    - Daily refresh: once per 24h per cover_key, replace the oldest/least-used variant with a new bg id
    - Watermark + AI badge always applied

    Backward compatibility:
    - Response still returns {coverKey, imageUrl, cached}
    - Firestore doc keeps imageUrl as a convenience pointer (last served)
    """

    brand = (req.brand or "").strip()
    model = (req.model or "").strip()
    if not brand or not model:
        raise HTTPException(status_code=400, detail="missing_brand_or_model")

    cover_key = _normalize_cover_key(req.cache_key) if req.cache_key else _cover_key(
        brand,
        model,
        req.body,
        req.year,
        req.color,
        req.generation,
    )

    doc_ref = _firebase_cover_doc(cover_key)
    doc = doc_ref.get()
    data = (doc.to_dict() or {}) if doc.exists else {}

    # Stored variants (list of dicts)
    variants = data.get("variants")
    if not isinstance(variants, list):
        variants = []

    # Legacy single url -> migrate into variants once
    legacy_url = data.get("imageUrl")
    if legacy_url and not variants:
        try:
            variants = [{"bg_id": 1, "url": str(legacy_url), "createdAt": data.get("createdAt") or datetime.utcnow().isoformat(), "useCount": 0}]
        except Exception:
            variants = [{"bg_id": 1, "url": str(legacy_url), "createdAt": datetime.utcnow().isoformat(), "useCount": 0}]

    regen_count = int(data.get("regenCount") or 0)
    last_served_bg = data.get("lastServedBgId")
    last_served_cat = data.get("lastServedCategory")
    last_refresh_at = data.get("lastRefreshAt")

    def _parse_iso(s: Any) -> Optional[datetime]:
        if not s or not isinstance(s, str):
            return None
        try:
            return datetime.fromisoformat(s.replace("Z", ""))
        except Exception:
            return None

    now = datetime.utcnow()
    refresh_due = False
    dt_last_refresh = _parse_iso(last_refresh_at)
    if dt_last_refresh is None:
        refresh_due = True if variants else False
    else:
        refresh_due = (now - dt_last_refresh).total_seconds() >= 24 * 3600

    # Helper choose bg id (category-aware: rotate background styles)
    # We keep 24 backgrounds, but avoid picking the same *style* repeatedly.
    BG_CATS: Dict[str, List[int]] = {
        "studio":   [1, 2, 3, 4, 5, 6],
        "urban":    [7, 8, 9, 10, 11, 12],
        "dramatic": [13, 14, 15, 16, 17, 18],
        "neutral":  [19, 20, 21, 22, 23, 24],
    }
    BG_ID_TO_CAT: Dict[int, str] = {bid: cat for cat, ids in BG_CATS.items() for bid in ids}

    def _bg_category(bid: int) -> str:
        try:
            return BG_ID_TO_CAT.get(int(bid), "studio")
        except Exception:
            return "studio"

    def _bg_ids_in_variants(vs: List[Dict[str, Any]]) -> List[int]:
        out: List[int] = []
        for x in vs:
            try:
                bid = int(x.get("bg_id"))
                if 1 <= bid <= 24:
                    out.append(bid)
            except Exception:
                continue
        return out

    def _choose_new_bg_id(
        existing_ids: List[int],
        vs: List[Dict[str, Any]],
        avoid_category: Optional[str],
    ) -> Tuple[int, str]:
        """Pick a new bg_id and its category.

        Goals:
        - Prefer bg_ids not already in variants
        - Prefer categories that are under-represented in the variant set
        - Avoid repeating last served category when possible
        """
        existing_set = set(existing_ids)

        # count current category usage
        counts: Dict[str, int] = {cat: 0 for cat in BG_CATS.keys()}
        for v in vs:
            try:
                counts[_bg_category(int(v.get("bg_id") or 1))] += 1
            except Exception:
                continue

        candidates: List[Tuple[int, str, List[int]]] = []
        for cat, ids in BG_CATS.items():
            avail = [i for i in ids if i not in existing_set]
            if avail:
                candidates.append((counts.get(cat, 0), cat, avail))

        # If we already used all 24 ids for this vehicle over time, allow reuse.
        if not candidates:
            for cat, ids in BG_CATS.items():
                candidates.append((counts.get(cat, 0), cat, list(ids)))

        # Avoid repeating the last served category if possible.
        if avoid_category:
            filtered = [c for c in candidates if c[1] != str(avoid_category)]
            if filtered:
                candidates = filtered

        # Prefer lowest-count categories to keep styles mixed.
        candidates.sort(key=lambda x: (x[0], x[1]))
        min_count = candidates[0][0]
        top = [c for c in candidates if c[0] == min_count]

        idx = int(uuid.uuid4().hex[:8], 16) % len(top)
        _, cat, avail = top[idx]
        idx2 = int(uuid.uuid4().hex[8:16], 16) % len(avail)
        return int(avail[idx2]), cat

    # Decide action
    action = "serve"
    if req.force_regenerate:
        # keep old safety: allow only once total for a key
        if regen_count >= 1 and variants:
            action = "serve"
        else:
            action = "refresh"
    else:
        if len(variants) < 3:
            action = "create"
        elif refresh_due and variants:
            action = "refresh"
        else:
            action = "serve"

    # Serve logic (no new gen)
    if action == "serve" and variants:
        # choose a variant, avoid repeating last bg if possible
        pick = None
        try:
            # prioritize those not equal to last served
            candidates = [v for v in variants if str(v.get("bg_id")) != str(last_served_bg)]
            # avoid repeating the same background *style* if possible
            try:
                if last_served_cat:
                    c2 = [v for v in candidates if _bg_category(int(v.get("bg_id") or 1)) != str(last_served_cat)]
                    if c2:
                        candidates = c2
            except Exception:
                pass
            if not candidates:
                candidates = variants
            idx = int(uuid.uuid4().hex[:8], 16) % len(candidates)
            pick = candidates[idx]
        except Exception:
            pick = variants[0]

        url = _fix_firebase_download_url(str(pick.get("url") or ""))
        bg_id = int(pick.get("bg_id") or 1)
        served_cat = _bg_category(bg_id)

        # update usage counters lightly (best-effort)
        try:
            for v in variants:
                if v is pick:
                    v["useCount"] = int(v.get("useCount") or 0) + 1
                    v["lastUsedAt"] = now.isoformat()
                    break
        except Exception:
            pass

        doc_ref.set(
            {
                "variants": variants,
                "imageUrl": url,
                "lastServedBgId": bg_id,
                "lastServedCategory": served_cat,
                "updatedAt": now.isoformat(),
            },
            merge=True,
        )

        return {"coverKey": cover_key, "imageUrl": url, "cached": True}

    # For create/refresh we need a new background id
    existing_ids = _bg_ids_in_variants(variants)
    new_bg_id, new_bg_cat = _choose_new_bg_id(
        existing_ids,
        variants,
        str(last_served_cat) if last_served_cat else None,
    )

    # If refresh: replace a slot (oldest / least used)
    replace_index = None
    if action == "refresh" and variants:
        # choose least used; tie-break by oldest createdAt
        def _score(v: Dict[str, Any]):
            use = int(v.get("useCount") or 0)
            created = _parse_iso(v.get("createdAt")) or datetime(1970, 1, 1)
            return (use, created)
        try:
            best_i = 0
            best = _score(variants[0])
            for i in range(1, len(variants)):
                sc = _score(variants[i])
                if sc < best:
                    best = sc
                    best_i = i
            replace_index = best_i
        except Exception:
            replace_index = 0

    # Generate a brand-recognizable image (no logos)
    try:
        img_bytes = _generate_vehicle_image_bytes(
            brand=brand,
            model=model,
            body=req.body,
            generation=req.generation,
            year=req.year,
            color=req.color,
        )
    except Exception as e:
        print("OpenAI image generation error:", repr(e))
        raise HTTPException(status_code=500, detail=f"image_generation_failed: {type(e).__name__}")

    # Watermark + AI badge + background variant
    final_bytes = _apply_watermark(
        img_bytes,
        watermark_text=os.getenv("COVER_WATERMARK_TEXT", "Oto Analiz"),
        bg_id=int(new_bg_id),
    )

    # Upload to Firebase Storage
    object_path = f"discover_covers/generated/{cover_key}/bg_{int(new_bg_id):02d}.webp"
    image_url = _firebase_upload_cover_bytes(cover_key, final_bytes, content_type="image/webp", object_path=object_path)

    # Update variants list
    entry = {
        "bg_id": int(new_bg_id),
        "category": str(new_bg_cat),
        "url": image_url,
        "createdAt": now.isoformat(),
        "lastUsedAt": now.isoformat(),
        "useCount": 1,
    }

    if action == "create":
        variants.append(entry)
    elif action == "refresh" and variants and replace_index is not None:
        variants[int(replace_index)] = entry
    else:
        # safety fallback
        variants = [entry]

    # Hard cap: keep at most 3 variants even under concurrent requests.
    try:
        def _score(v: Dict[str, Any]):
            use = int(v.get("useCount") or 0)
            created = _parse_iso(v.get("createdAt")) or datetime(1970, 1, 1)
            return (use, created)
        while len(variants) > 3:
            # drop the least used / oldest
            drop_i = 0
            best = _score(variants[0])
            for i in range(1, len(variants)):
                sc = _score(variants[i])
                if sc < best:
                    best = sc
                    drop_i = i
            variants.pop(drop_i)
    except Exception:
        variants = variants[-3:]

    # regen counter (only for force_regenerate)
    next_regen_count = regen_count
    if req.force_regenerate:
        next_regen_count = regen_count + 1

    doc_ref.set(
        {
            "brand": brand,
            "model": model,
            "body": req.body,
            "generation": req.generation,
            "year": req.year,
            "color": req.color,
            "variants": variants,
            "imageUrl": image_url,
            "lastServedBgId": int(new_bg_id),
            "lastServedCategory": str(new_bg_cat),
            "lastRefreshAt": now.isoformat(),
            "regenCount": next_regen_count,
            "createdAt": data.get("createdAt") or now.isoformat(),
            "updatedAt": now.isoformat(),
        },
        merge=True,
    )

    return {"coverKey": cover_key, "imageUrl": image_url, "cached": False, "bgId": int(new_bg_id), "action": action}

@app.post("/analyze")
async def analyze(req: AnalyzeRequest) -> Dict[str, Any]:
    # Normal = Hızlı Analiz (default: deterministic, hızlı, ucuz)
    use_llm = (os.getenv("USE_LLM_NORMAL", "0").strip().lower() in ("1", "true", "yes", "y"))
    if use_llm:
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
    return quick_analyze_impl(req)


@app.post("/premium_analyze")
async def premium_analyze(req: AnalyzeRequest) -> Dict[str, Any]:
    """
    Premium analiz:
    - USE_LLM_PREMIUM=1 ise: Premium metin/kartlar TAMAMEN LLM ile üretilir (deterministik veriler referans olarak verilir).
    - Aksi halde: deterministik template (fallback).
    """
    use_llm = (os.getenv("USE_LLM_PREMIUM", "0").strip().lower() in ("1", "true", "yes", "y"))
    llm_used = False

    # Fallback her zaman hazır dursun
    fallback = premium_analyze_impl(req)

    if use_llm:
        try:
            print("USE_LLM_PREMIUM=1 -> premium_full_llm START")
            enriched = build_enriched_context(req)

            payload = {
                "vehicle": req.vehicle.dict(),
                "profile": req.profile.dict() if req.profile else {},
                "ad_description": req.ad_description,
                "context": req.context or {},
                "enriched": enriched,
                "fixed_scores": fallback.get("scores"),
                "fixed_preview": fallback.get("preview"),
            }

            out = call_llm_json(
                OPENAI_MODEL_PREMIUM,
                SYSTEM_PROMPT_PREMIUM_FULL,
                json.dumps(payload, ensure_ascii=False),
            )

            # Şema doğrulama (kırılmasın)
            if isinstance(out, dict) and isinstance(out.get("cards"), list) and isinstance(out.get("scores"), dict):
                out["meta"] = {"use_llm_premium": True, "llm_used": True}
                llm_used = True
                print("USE_LLM_PREMIUM=1 -> premium_full_llm END (ok)")
                return out

            print("premium_full_llm invalid output -> fallback")
        except Exception as e:
            print("premium_full_llm error -> fallback:", e)

    # fallback
    fallback["meta"] = {"use_llm_premium": use_llm, "llm_used": llm_used}
    return fallback


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
    """
    Compare endpoint now supports **mode switching**:
    - quick/fast/normal  -> uses quick_analyze_impl (Hızlı Analiz)
    - premium           -> uses premium_analyze_impl (Detaylı Premium)
    Flutter tarafında switch ile şu alanlardan birini göndermen yeterli:
      - req.analysis_mode = "quick" | "premium"
      - veya profile içinde: {"analysis_mode":"premium"} / {"mode":"premium"} / {"is_premium": true}
    """
    # -------------------------
    # Resolve mode
    # -------------------------
    mode = None
    try:
        mode = getattr(req, "analysis_mode", None)
    except:
        mode = None

    prof_obj = req.profile or Profile()
    try:
        # profile extra fields also supported (Config extra=allow)
        mode = mode or getattr(prof_obj, "analysis_mode", None) or getattr(prof_obj, "mode", None)
        is_premium = getattr(prof_obj, "is_premium", None)
        if mode is None and isinstance(is_premium, bool):
            mode = "premium" if is_premium else "quick"
    except:
        pass

    mode_s = str(mode or "quick").strip().lower()
    if mode_s in ("premium", "pro", "detailed"):
        mode_s = "premium"
    elif mode_s in ("quick", "fast", "normal", "hizli", "hızlı"):
        mode_s = "quick"
    else:
        mode_s = "quick"

    # -------------------------
    # Build per-side AnalyzeRequest
    # -------------------------
    left_req = AnalyzeRequest(
        profile=prof_obj,
        vehicle=req.left.vehicle,
        ad_description=req.left.ad_description,
        screenshots_base64=req.left.screenshots_base64,
        context={"compare_side": "left"},
    )
    right_req = AnalyzeRequest(
        profile=prof_obj,
        vehicle=req.right.vehicle,
        ad_description=req.right.ad_description,
        screenshots_base64=req.right.screenshots_base64,
        context={"compare_side": "right"},
    )

    # -------------------------
    # Produce reports
    # -------------------------
    if mode_s == "premium":
        left_report = premium_analyze_impl(left_req)
        right_report = premium_analyze_impl(right_req)
    else:
        left_report = quick_analyze_impl(left_req)
        right_report = quick_analyze_impl(right_req)

    def _overall(rep: Dict[str, Any]) -> int:
        try:
            return int(((rep.get("scores") or {}).get("overall_100")) or 0)
        except:
            return 0

    left_overall = _overall(left_report)
    right_overall = _overall(right_report)

    # tie-breaker: personal_fit if exists (premium), else economy_100 (quick uses economy_100=parts_service_100)
    def _tiebreak(rep: Dict[str, Any]) -> int:
        sc = rep.get("scores") or {}
        for k in ("personal_fit_100", "economy_100", "mechanical_100", "body_100"):
            v = sc.get(k)
            if isinstance(v, int):
                return v
        return 0

    if left_overall > right_overall:
        better = "left"
    elif right_overall > left_overall:
        better = "right"
    else:
        better = "left" if _tiebreak(left_report) >= _tiebreak(right_report) else "right"

    # -------------------------
    # Optional LLM compare summary (keeps old behavior, but now includes reports)
    # -------------------------
    left_v = req.left.vehicle
    right_v = req.right.vehicle

    payload = {
        "mode": mode_s,
        "left": {
            "vehicle": left_v.dict(),
            "ad_description": req.left.ad_description or "",
            "scores": left_report.get("scores", {}),
            "summary": left_report.get("summary", {}),
        },
        "right": {
            "vehicle": right_v.dict(),
            "ad_description": req.right.ad_description or "",
            "scores": right_report.get("scores", {}),
            "summary": right_report.get("summary", {}),
        },
        "profile": prof_obj.dict(),
        "rule": "Bütçe/kullanım amacı/şehir içi-uzun yol gibi profil verilerini dikkate al; kesin hüküm verme; Türkçe ve net yaz."
    }

    out = call_llm_json(OPENAI_MODEL_COMPARE, SYSTEM_PROMPT_COMPARE, json.dumps(payload, ensure_ascii=False))

    if isinstance(out, dict):
        # ensure winner aligns with deterministic scores if model returns something else
        out["better_overall"] = out.get("better_overall") or better
        out["mode"] = mode_s
        out["left_report"] = left_report
        out["right_report"] = right_report
        out["left_overall_100"] = left_overall
        out["right_overall_100"] = right_overall
        return out

    # deterministic fallback (no LLM)
    left_title = f"{left_v.year or ''} {left_v.make} {left_v.model}".strip()
    right_title = f"{right_v.year or ''} {right_v.make} {right_v.model}".strip()

    def _short(rep: Dict[str, Any]) -> str:
        s = rep.get("summary") or {}
        if isinstance(s, dict):
            return (s.get("short_comment") or "").strip()
        return ""

    summary = (
        f"Karşılaştırma modu: **{('Premium' if mode_s=='premium' else 'Hızlı')}**. "
        f"Genel skor: Sol **{left_overall}/100**, Sağ **{right_overall}/100**. "
        f"Öneri (genel): **{('Sol' if better=='left' else 'Sağ')}** taraf daha dengeli görünüyor."
    )

    return {
        "mode": mode_s,
        "better_overall": better,
        "summary": summary,
        "left_pros": (left_report.get("summary", {}) or {}).get("pros", []) if isinstance(left_report.get("summary"), dict) else [],
        "left_cons": (left_report.get("summary", {}) or {}).get("cons", []) if isinstance(left_report.get("summary"), dict) else [],
        "right_pros": (right_report.get("summary", {}) or {}).get("pros", []) if isinstance(right_report.get("summary"), dict) else [],
        "right_cons": (right_report.get("summary", {}) or {}).get("cons", []) if isinstance(right_report.get("summary"), dict) else [],
        "use_cases": {
            "family_use": "Aile kullanımı için iç hacim/konfor ve masraf bandı birlikte düşünülmeli.",
            "long_distance": "Uzun yolda tüketim + bakım disiplini belirleyicidir; test sürüşü ve servis kaydı kritik.",
            "city_use": "Şehir içinde vites tipi ve yakıt tercihi (özellikle dizel/DPF) daha kritik hale gelir.",
        },
        "left_report": left_report,
        "right_report": right_report,
        "left_overall_100": left_overall,
        "right_overall_100": right_overall,
        "left_title": left_title,
        "right_title": right_title,
    }


@app.post("/otobot_legacy")
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


# ============================
# YOL-1 FINAL OVERRIDE (FRONT/NET + MODEL-CORRECT)
# ============================
# NOTE: Python uses the *last* definition, so this overrides earlier helpers safely.

from typing import Iterable
from PIL import ImageFilter

_FRONT_HINTS = [
    "front", "front view", "front-angle", "frontal", "three quarter", "3/4", "three-quarter", "angle view",
    "side", "side view",
]
_REAR_BAD = [
    "rear", "back view", "back", "taillight", "tail light", "behind", "rear view",
]
_SCENE_BAD = [
    "interior", "inside", "dashboard", "cockpit", "steering", "seat", "console", "gear",
    "engine", "close up", "close-up", "detail", "rim", "tire", "tyre",
    "showroom", "dealership", "parking", "traffic", "fleet",
    "night", "dark", "low light", "low-light",
    "door open", "open door", "trunk open", "boot open",
]
_MULTI_BAD = ["cars", "two cars", "three cars", "multiple cars", "many cars"]

def _img_is_sharp_and_bright(img: "Image.Image") -> bool:
    """
    Cheap 'net' check:
    - brightness: mean grayscale must be above a threshold
    - sharpness: variance of edge map must be above a threshold
    """
    # downscale for speed
    im = img.convert("L")
    im = im.resize((320, int(320 * im.height / max(im.width, 1))), Image.Resampling.BILINEAR)
    # brightness
    px = list(im.getdata())
    if not px:
        return False
    mean = sum(px) / len(px)
    if mean < 55:  # too dark
        return False

    # sharpness via edge variance
    edges = im.filter(ImageFilter.FIND_EDGES)
    ex = list(edges.getdata())
    if not ex:
        return False
    emean = sum(ex) / len(ex)
    var = sum((v - emean) ** 2 for v in ex) / len(ex)
    # Typical sharp photos will be well above this.
    return var >= 120.0

def _norm_brand_model_for_match(brand: str, model: str) -> Tuple[str, str, List[str], List[str]]:
    """
    Returns:
      b: normalized brand token
      m: normalized model token (cleaned)
      must_phrases: phrases that MUST appear in alt (lowercase)
      must_tokens: tokens any of which must appear in alt (lowercase)
    """
    def clean(x: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s\-]", " ", (x or "").lower())).strip()

    brand_alias = {
        "mercdes": "mercedes",
        "mercedes-benz": "mercedes",
        "benz": "mercedes",
        "wolkswagen": "vw",
        "volkswagen": "vw",
        "bmv": "bmw",
        "hyundai ": "hyundai",
    }
    b = clean(brand)
    b = brand_alias.get(b, b)

    m_raw = clean(model)

    must_phrases: List[str] = []
    must_tokens: List[str] = []

    # Mercedes class hard handling: "c class" is NOT the same as "s class"
    if b in ("mercedes", "mb"):
        # Normalize variants
        m_raw = m_raw.replace("cclass", "c class").replace("c-class", "c class")
        m_raw = m_raw.replace("sclass", "s class").replace("s-class", "s class")
        m_raw = m_raw.replace("eclass", "e class").replace("e-class", "e class")
        m_raw = m_raw.replace("aclass", "a class").replace("a-class", "a class")
        m_raw = m_raw.replace("bclass", "b class").replace("b-class", "b class")

        # If user says "C Class" or just "C"
        if m_raw in ("c", "c class"):
            must_phrases = ["c class", "c-class", "cclass"]
            must_tokens = ["mercedes", "benz"]
            return b, "c class", must_phrases, must_tokens
        if m_raw in ("e", "e class"):
            must_phrases = ["e class", "e-class", "eclass"]
            must_tokens = ["mercedes", "benz"]
            return b, "e class", must_phrases, must_tokens
        if m_raw in ("s", "s class"):
            must_phrases = ["s class", "s-class", "sclass"]
            must_tokens = ["mercedes", "benz"]
            return b, "s class", must_phrases, must_tokens

    # General case:
    toks = [t for t in re.split(r"[\s\-]+", m_raw) if t]
    if len(m_raw) >= 3:
        # require the full phrase if multi-token
        if len(toks) >= 2:
            must_phrases.append(m_raw)
        # pick strongest tokens (avoid generic)
        generic = {"class", "series", "model", "car", "auto"}
        strong = [t for t in toks if t not in generic and len(t) >= 3]
        if strong:
            must_tokens.extend(strong[:2])
        else:
            must_tokens.extend([t for t in toks if len(t) >= 2][:2])

    return b, m_raw, must_phrases, must_tokens

def _alt_text(p: Dict[str, Any]) -> str:
    return (p.get("alt") or "").lower().strip()

def _passes_front_only(alt: str) -> bool:
    if any(w in alt for w in _REAR_BAD):
        return False
    if any(w in alt for w in _SCENE_BAD):
        return False
    if any(w in alt for w in _MULTI_BAD):
        return False
    if not any(h in alt for h in _FRONT_HINTS):
        return False
    return True

def _passes_brand_model(alt: str, b: str, must_phrases: List[str], must_tokens: List[str]) -> bool:
    if b == "vw":
        if "vw" not in alt and "volkswagen" not in alt:
            return False
    elif b == "mercedes":
        if "mercedes" not in alt and "benz" not in alt:
            return False
    else:
        if b and b not in alt:
            return False

    if must_phrases:
        if not any(ph in alt for ph in must_phrases):
            return False
        # prevent C-class confusing with S/E/A/B/G
        if any(ph in must_phrases for ph in ["c class", "c-class", "cclass"]):
            if any(x in alt for x in [" s class", " s-class", "sclass", " e class", "eclass", " a class", "aclass", " b class", "bclass", " g class", "gclass"]):
                return False
    if must_tokens:
        if not any(t in alt for t in must_tokens):
            return False
    return True

# ================================
# Premium follow-up question (1 hak)
# ================================

class PremiumQuestionRequest(BaseModel):
    analysis_text: str
    question: str

class PremiumQuestionResponse(BaseModel):
    answer: str

@app.post("/premium_question", response_model=PremiumQuestionResponse)
async def premium_question(req: PremiumQuestionRequest):
    if OpenAI is None:
        raise HTTPException(status_code=500, detail="OpenAI client not available")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)

    prompt = f"""
Aşağıda bir araç için hazırlanmış premium analiz raporu var.
Kullanıcı bu analizle ilgili bir soru soruyor.

ANALİZ:
{req.analysis_text}

SORU:
{req.question}

Kısa, net ve kullanıcı dostu şekilde cevap ver.
"""

    try:
        resp = client.responses.create(
            model=os.environ.get("SS_OCR_MODEL", "gpt-5-mini"),
            input=prompt,
        )
        answer = resp.output_text
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# Optimized Premium Question Endpoint (Short & Fast)
# =========================

@app.post("/premium_question")
async def premium_question(payload: dict):
    analysis_text = payload.get("analysis_text", "")[:6000]  # limit size
    question = payload.get("question", "")

    prompt = f"""
Kısa ve öz cevap ver (max 6-8 cümle).

Analiz özeti:
{analysis_text}

Soru:
{question}
"""

    try:
        response = client.chat.completions.create(
            model=os.getenv("PREMIUM_QUESTION_MODEL", "gpt-5-mini"),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=280,
            temperature=0.4,
        )

        answer = response.choices[0].message.content.strip()
        return {"answer": answer}

    except Exception as e:
        return {"error": str(e)}
