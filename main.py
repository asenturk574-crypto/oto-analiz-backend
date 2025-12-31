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

    # Flutter tarafı bazen "text" gönderiyor, ekstra alanları kabul et
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

    # Ayrıca extra alanlarda gelebilir
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
# 1) SEGMENT PROFİLLERİ (TR piyasasına uygun "referans" aralıklar)
#    -> Uçuk olmayı engelleyen temel çerçeve burası
# ---------------------------------------------------------
SEGMENT_PROFILES: Dict[str, Dict[str, Any]] = {
    "B_HATCH": {
        "name": "B segment (küçük hatch)",
        "maintenance_yearly_range": (12000, 25000),
        "insurance_level": "orta",
        "parts": "çok kolay",
        "resale": "hızlı",
        "notes": [
            "Parça/usta kolay bulunur, şehir içinde pratik.",
            "Doğru bakım olursa masraflar genelde kontrol edilebilir."
        ],
    },
    "C_SEDAN": {
        "name": "C segment (aile sedan/hatch)",
        "maintenance_yearly_range": (15000, 32000),
        "insurance_level": "orta",
        "parts": "kolay",
        "resale": "hızlı-orta",
        "notes": [
            "Türkiye’de en likit segmentlerden, ikinci elde alıcı bulunur.",
            "Bakım maliyeti B segmente göre biraz daha yüksektir."
        ],
    },
    "C_SUV": {
        "name": "C segment SUV",
        "maintenance_yearly_range": (18000, 38000),
        "insurance_level": "orta-yüksek",
        "parts": "orta-kolay",
        "resale": "orta-hızlı",
        "notes": [
            "SUV olduğu için lastik/fren/alt takım giderleri biraz daha yüksek olabilir.",
            "Konfor ve aile kullanımı güçlü, ama tüketim ve vergi etkisi unutulmamalı."
        ],
    },
    "D_SEDAN": {
        "name": "D segment (konfor sedan)",
        "maintenance_yearly_range": (22000, 50000),
        "insurance_level": "orta-yüksek",
        "parts": "orta",
        "resale": "orta",
        "notes": [
            "Konfor yüksek, fakat parça/bakım kalemleri C segmente göre pahalılaşır."
        ],
    },
    "PREMIUM_D": {
        "name": "Premium D segment",
        "maintenance_yearly_range": (32000, 75000),
        "insurance_level": "yüksek",
        "parts": "orta-zor",
        "resale": "orta",
        "notes": [
            "Premium sınıfta bakım, işçilik ve parça maliyeti belirgin artar.",
            "Kasko/sigorta fiyatları kullanıcı profiline göre ciddi değişebilir."
        ],
    },
    "E_SEGMENT": {
        "name": "E segment / üst sınıf",
        "maintenance_yearly_range": (45000, 120000),
        "insurance_level": "çok yüksek",
        "parts": "zor",
        "resale": "yavaş-orta",
        "notes": [
            "Masraf kalemleri yüksek: lastik, fren, yağ, şanzıman/aktarma vb.",
        ],
    },
}


# ---------------------------------------------------------
# 2) ANCHOR MODEL VERİTABANI (bol veri)
#    -> Model yoksa: aynı segment anchor’lardan tahmin
# ---------------------------------------------------------
ANCHORS: List[Dict[str, Any]] = [
    # --- B Segment ---
    {
        "key": "renault clio",
        "segment": "B_HATCH",
        "aliases": ["clio", "clio 4", "clio 5"],
        "chronic": ["Ön takım burç/salıncak yıpranması (km+yaşla)", "Turbo dizellerde EGR/DPF hassasiyeti (şehir içi)"],
        "parts": "çok kolay",
        "resale": "hızlı",
    },
    {"key": "vw polo", "segment": "B_HATCH", "aliases": ["polo"], "chronic": ["DSG varsa bakım geçmişi kritik"], "parts": "kolay", "resale": "hızlı-orta"},
    {"key": "hyundai i20", "segment": "B_HATCH", "aliases": ["i20"], "chronic": ["Genelde düşük kronik, bakım standardı önemli"], "parts": "kolay", "resale": "orta-hızlı"},

    # --- C Sedan / Hatch ---
    {"key": "toyota corolla", "segment": "C_SEDAN", "aliases": ["corolla"], "chronic": ["Genelde düşük kronik, düzenli bakım önemli"], "parts": "kolay", "resale": "hızlı"},
    {"key": "honda civic", "segment": "C_SEDAN", "aliases": ["civic"], "chronic": ["LPG varsa montaj/ayar ve subap sağlığı takip"], "parts": "kolay", "resale": "hızlı"},
    {"key": "renault megane", "segment": "C_SEDAN", "aliases": ["megane"], "chronic": ["1.5 dCi’de DPF/EGR (şehir içi)", "EDC varsa yağ/bakım geçmişi önemli"], "parts": "kolay", "resale": "hızlı-orta"},
    {"key": "fiat egea", "segment": "C_SEDAN", "aliases": ["egea", "tipo"], "chronic": ["Multijet dizelde DPF (şehir içi)", "Trim sesleri (kişiden kişiye)"], "parts": "çok kolay", "resale": "hızlı"},
    {"key": "vw passat", "segment": "D_SEDAN", "aliases": ["passat"], "chronic": ["DSG varsa bakım geçmişi kritik", "TDI’de DPF/EGR"], "parts": "orta", "resale": "orta-hızlı"},
    {"key": "skoda superb", "segment": "D_SEDAN", "aliases": ["superb"], "chronic": ["DSG geçmişi önemli"], "parts": "orta", "resale": "orta"},

    # --- C SUV ---
    {"key": "nissan qashqai", "segment": "C_SUV", "aliases": ["qashqai"], "chronic": ["CVT varsa bakım geçmişi kritik"], "parts": "orta", "resale": "orta-hızlı"},
    {"key": "kia sportage", "segment": "C_SUV", "aliases": ["sportage"], "chronic": ["Dizelde DPF/EGR (şehir içi)"], "parts": "orta", "resale": "orta"},
    {"key": "hyundai tucson", "segment": "C_SUV", "aliases": ["tucson"], "chronic": ["Dizelde DPF/EGR (şehir içi)"], "parts": "orta", "resale": "orta"},

    # --- Premium D ---
    {"key": "bmw 320", "segment": "PREMIUM_D", "aliases": ["320i", "320d", "3.20", "3.20i", "3.20d", "3 series", "3 serisi"],
     "chronic": ["Zamanla burç/ön takım", "Turbo-dizelde EGR/DPF (şehir içi)", "Zincir/yağ bakımı ihmal edilirse risk büyür"],
     "parts": "orta-zor", "resale": "orta"},
    {"key": "mercedes c200", "segment": "PREMIUM_D", "aliases": ["c180", "c200", "c220", "c class", "c serisi"],
     "chronic": ["Elektronik/konfor donanımlarında yaşa bağlı arızalar", "Bakım/yağ standardı kritik"],
     "parts": "orta", "resale": "orta"},
    {"key": "audi a4", "segment": "PREMIUM_D", "aliases": ["a4", "a4 allroad", "allroad"],
     "chronic": ["DSG/S-tronic geçmişi önemli", "TFSI’de yağ tüketimi geçmişi (bazı nesiller)", "Quattro bakım yükü artabilir"],
     "parts": "orta-zor", "resale": "orta"},
    {"key": "volvo s60", "segment": "PREMIUM_D", "aliases": ["s60"], "chronic": ["Parça/servis maliyeti premium seviyede"], "parts": "zor", "resale": "orta-yavaş"},
]

def detect_segment(make: str, model: str) -> str:
    s = _norm(f"{make} {model}")

    # Premium markalar öncelik
    if any(k in s for k in ["bmw", "mercedes", "audi", "volvo", "lexus", "range rover"]):
        return "PREMIUM_D"

    # Basit heuristik (sen zaten geliştirdin, bunu güçlendirdim)
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
        # marka/model alias eşleşmesi
        aliases = a.get("aliases") or []
        key = _norm(a.get("key", ""))
        if key and key in target:
            score += 10
        for al in aliases:
            if _norm(al) in target:
                score += 8

        # marka benzerliği
        if any(m in target for m in ["bmw", "mercedes", "audi", "volvo"]) and any(m in key for m in ["bmw", "mercedes", "audi", "volvo"]):
            score += 2

        if score > 0:
            scored.append((score, a))

    scored.sort(key=lambda x: x[0], reverse=True)

    if scored:
        return [x[1] for x in scored[:limit]]

    # hiç eşleşme yoksa: segmentten 2-3 anchor al
    fallback = [a for a in ANCHORS if a.get("segment") == segment][:limit]
    return fallback


# ---------------------------------------------------------
# 3) UÇUK OLMAYAN "MALİYET" HESABI + CLAMP
# ---------------------------------------------------------
def estimate_costs(req: AnalyzeRequest) -> Dict[str, Any]:
    v = req.vehicle
    p = req.profile or Profile()
    segment_code = detect_segment(v.make, v.model)
    seg = SEGMENT_PROFILES.get(segment_code, SEGMENT_PROFILES["C_SEDAN"])

    listed_price = _parse_listed_price(req)

    # yaş / km
    age = None
    if v.year:
        age = max(0, 2025 - v.year)
    mileage = int(v.mileage_km or 0)

    base_min, base_max = seg["maintenance_yearly_range"]

    # çarpanlar
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

    # kullanım + yakıt
    usage_mult = 1.0
    if p.usage == "city":
        usage_mult = 1.15  # şehir içi daha yıpratır
    elif p.usage == "highway":
        usage_mult = 0.95

    fuel_mult = 1.0
    fuel_risk = "orta"
    fuel = (v.fuel or p.fuel_preference or "").lower()

    if fuel == "diesel":
        fuel_mult = 1.05
        if p.usage == "city" and mileage >= 120_000:
            fuel_risk = "orta-yüksek (DPF/EGR/enjektör riski şehir içiyle artar)"
    elif fuel == "lpg":
        fuel_mult = 0.95
        fuel_risk = "orta (montaj/ayar ve subap sağlığı önemli)"
    elif fuel in ("hybrid", "electric"):
        fuel_mult = 0.9
        fuel_risk = "düşük-orta (batarya sağlığına bağlı)"

    # bakım tahmini
    est_min = int(base_min * age_mult * km_mult * usage_mult * fuel_mult)
    est_max = int(base_max * age_mult * km_mult * usage_mult * fuel_mult)

    # --- UÇUK DEĞER ENGELİ (CLAMP) ---
    # Eğer fiyat varsa: bakım yıllık -> aracın fiyatının %1.5 - %10 aralığında kalmaya zorla
    # (çok kaba ama "300k araca 70k bakım" gibi uçukları keser)
    if listed_price and listed_price > 100000:
        hard_min = int(listed_price * 0.015)  # 1.5%
        hard_max = int(listed_price * 0.10)   # 10%
        est_min = max(est_min, hard_min)
        est_max = min(est_max, hard_max)

        # ayrıca segment bazlı mutlak tavan
        if segment_code in ("B_HATCH", "C_SEDAN"):
            est_max = min(est_max, 55000)
        elif segment_code in ("C_SUV", "D_SEDAN"):
            est_max = min(est_max, 80000)
        elif segment_code == "PREMIUM_D":
            est_max = min(est_max, 140000)

        if est_min > est_max:
            est_min = int(est_max * 0.75)

    # yakıt yıllık (çok kaba): profil km’ye göre
    # burada da uçuk olmasın diye bir aralık veriyoruz
    # (LLM zaten metni yazarken bunu "tahmini" kullanacak)
    km_year = max(0, int(p.yearly_km or 15000))
    # segment bazlı tüketim bandı (L/100)
    cons_band = {
        "B_HATCH": (5.5, 7.5),
        "C_SEDAN": (6.5, 9.0),
        "C_SUV": (7.5, 11.0),
        "D_SEDAN": (7.5, 11.0),
        "PREMIUM_D": (8.0, 12.5),
        "E_SEGMENT": (9.0, 15.0),
    }.get(segment_code, (6.5, 9.5))

    return {
        "segment_code": segment_code,
        "segment_name": seg["name"],
        "age": age,
        "mileage_km": mileage,
        "listed_price_try": listed_price,
        "maintenance_yearly_try_min": est_min,
        "maintenance_yearly_try_max": est_max,
        "insurance_level": seg["insurance_level"],
        "parts_availability": seg["parts"],
        "resale_speed": seg["resale"],
        "fuel_risk_comment": fuel_risk,
        "consumption_l_per_100km_band": cons_band,
        "segment_notes": seg.get("notes", []),
    }


def build_enriched_context(req: AnalyzeRequest) -> Dict[str, Any]:
    v = req.vehicle
    seg_info = estimate_costs(req)
    anchors = find_anchor_matches(v.make, v.model, seg_info["segment_code"], limit=3)

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
            "consumption_l_per_100km_band": seg_info["consumption_l_per_100km_band"],
        },
        "risk": {
            "fuel_risk_comment": seg_info["fuel_risk_comment"],
            "age": seg_info["age"],
            "mileage_km": seg_info["mileage_km"],
        },
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
            "Donanım/hasar/tramer/boya-değişen gibi detayları varsa bunları dikkate al."
        )

    if not (v.make.strip() or v.model.strip() or ad_text or all_ss):
        ad_text = "Kullanıcı çok az bilgi verdi. Türkiye ikinci el piyasasında genel kabul gören kriterlerle, varsayımsal bir değerlendirme yap."

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

--- Referans veri + segment emsal (BU KISMI TEMEL AL) ---
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
            "short_comment": "Sınırlı bilgiye göre genel değerlendirme yapıldı.",
            "pros": ["Ekspertiz ve tramer ile netleşebilir.", "Türkiye piyasasında yaygın kriterlerle nötr bir analiz."],
            "cons": ["Detaylar için ilan açıklaması/SS/ekspertiz gerekli.", "Bakım geçmişi teyit edilmeden karar verilmemeli."],
            "estimated_risk_level": "orta"
        },
        "preview": {
            "title": title,
            "price_tag": None,
            "spoiler": "Genel değerlendirme hazır. Ekspertiz ve bakım geçmişi teyit edilmeden karar verme.",
            "bullets": [
                "Tramer/hasar kontrolü",
                "Bakım kayıtları & km uyumu",
                "Lastik/fren/alt takım kontrolü"
            ]
        },
        "result": "Genel bir değerlendirme üretildi. Detay için ilan açıklaması ve ekspertiz önerilir."
    }


def fallback_premium(req: AnalyzeRequest) -> Dict[str, Any]:
    v = req.vehicle
    title = f"{v.year or ''} {v.make} {v.model}".strip() or "Premium Analiz"
    ctx = build_enriched_context(req)

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
            "yearly_maintenance_tr": ctx["costs"]["maintenance_yearly_try_min"],
            "yearly_fuel_tr": 0,
            "insurance_level": ctx["market"]["insurance_level"],
            "notes": "Bakım aralığı segment + yaş + km + kullanım profiline göre tahmini verilmiştir."
        },
        "risk_analysis": {
            "chronic_issues": [x for a in ctx["anchors_used"] for x in (a.get("chronic") or [])][:6],
            "risk_level": "orta",
            "warnings": [
                "Ekspertiz + tramer teyidi olmadan kesin karar verme.",
                "Bakım geçmişi/yağ değişim düzeni kritik."
            ]
        },
        "summary": {
            "short_comment": "Segment emsali ve kullanıcı profiline göre premium özet üretildi.",
            "pros": ["Parça/servis durumu segmentine göre değerlendirildi.", "Kullanım profiline göre uyarılar eklendi."],
            "cons": ["Birebir model verisi sınırlı olabilir, emsal segmentten tahmin yürütülür."],
            "who_should_buy": "Bütçesini bilen, ekspertiz yaptıracak ve bakım planı oluşturacak kullanıcılar."
        },
        "preview": {
            "title": title,
            "price_tag": None,
            "spoiler": "Premium formatta, segment emsal ve kişisel profil odaklı özet.",
            "bullets": [
                "Bakım/masraf aralığı verildi",
                "Sigorta-kasko riski yorumlandı",
                "Yedek parça & ikinci el likidite notu"
            ]
        },
        "details": {
            "personal_fit": [
                "İlk araç/öğrenci isen kasko/sigorta primi bütçeyi zorlayabilir (özellikle premium segmentte).",
                "Şehir içi kullanım dizelde DPF/EGR riskini artırabilir; kullanım tipine göre karar ver."
            ],
            "parts_and_service": [
                f"Parça bulunabilirliği: {ctx['market']['parts_availability']}",
                f"İkinci el likidite: {ctx['market']['resale_speed']}"
            ],
        },
        "result": "Premium değerlendirme (fallback) üretildi. Daha dolu sonuç için ilan açıklaması ve SS içeriği önerilir."
    }


def fallback_manual(req: AnalyzeRequest) -> Dict[str, Any]:
    return fallback_normal(req)


def fallback_compare(req: CompareRequest) -> Dict[str, Any]:
    left_title = (f"{req.left.vehicle.make} {req.left.vehicle.model}").strip() or "Sol araç"
    right_title = (f"{req.right.vehicle.make} {req.right.vehicle.model}").strip() or "Sağ araç"

    return {
        "better_overall": "left",
        "summary": f"{left_title} genel kullanım için daha dengeli varsayıldı. Ancak iki araç için de ekspertiz + tramer şart.",
        "left_pros": [f"{left_title} için maliyet/performans dengesi daha iyi olabilir.", "Aile/karışık kullanım için uygun olabilir."],
        "left_cons": ["Gerçek durum için yerinde inceleme gerekir."],
        "right_pros": [f"{right_title} doğru bakımla mantıklı olabilir."],
        "right_cons": ["Masraf kalemleri ve riskleri daha dikkatli incelenmeli."],
        "use_cases": {
            "family_use": f"Aile kullanımı için {left_title} daha mantıklı varsayıldı.",
            "long_distance": "Her iki araç da düzenli bakım ile uzun yolda kullanılabilir.",
            "city_use": "Şehir içi tüketim, vites tipi ve bakım geçmişi belirleyici."
        }
    }


def fallback_otobot(question: str) -> Dict[str, Any]:
    return {
        "answer": "Bütçe, yıllık km, kullanım tipi ve beklentilere göre segment seçmek en doğrusu. Yıllık km yüksekse dizel/ekonomik benzinli, düşükse benzinli/hybrid daha mantıklı olabilir. Satın almadan önce ekspertiz ve tramer şart.",
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
- Tüm alanlar mutlaka olsun.
- PREVIEW Keşfet için:
  - 'alınır/alınmaz/sakın/tehlikeli' KULLANMA.
  - Fiyat rakamı yazma; sadece 'Uygun/Normal/Yüksek' etiketi veya null.
- result: 6-10 cümle arası, madde + kısa paragraf karışık olabilir.
- Dil: Türkçe.
"""

SYSTEM_PROMPT_PREMIUM = """
Sen 'Oto Analiz' uygulamasının PREMIUM analiz asistanısın.
Elindeki "Referans veri + segment emsal" bloğunu temel al.
Model yoksa aynı segmentteki "anchors_used" verileriyle tahmin yürüt; bunu açıkça belirt.
Uçuk rakam yazma: bakım maliyeti "costs.maintenance_yearly_try_min/max" bandının dışına taşmasın.

ÇIKTIYI SADECE GEÇERLİ BİR JSON OLARAK DÖN. ŞABLON:

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
    "yearly_fuel_tr": 0,
    "insurance_level": "orta",
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

Premium Kurallar:
- "result" PREMIUM hissi vermeli: kısa başlıklar + madde madde; 25-45 satır arası hedefle (çok kısaltma).
- "details" içini DOLDUR: en az 5 başlıkta en az 3 madde olsun.
- Sigorta/kasko: kullanıcı profiline göre yorum yap (ilk araç, öğrenci, yaş, şehir, yıllık km).
- Yedek parça/usta: segment+anchor bazlı net cümle kur.
- Satış/likidite: segment+model algısına göre.
- Bakım: "min-max bandı" ver, uçurma.
- PREVIEW Keşfet için:
  - 'alınır/alınmaz/sakın/tehlikeli' KULLANMA.
  - Fiyat rakamı yazma; sadece 'Uygun/Normal/Yüksek' veya null.
- Dil: Türkçe, sadece JSON.
"""

SYSTEM_PROMPT_MANUAL = """
Sen 'Oto Analiz' uygulamasında KULLANICININ KENDİ ARACI için manuel analiz yapan asistansın.
Çıktı formatın NORMAL analizle aynıdır.

Kurallar:
- PREVIEW nötr.
- Bilgi azsa bile genel ve faydalı öneriler ver.
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
# MANUEL / KENDİ ARAÇ ANALİZİ
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
        raise HTTPException(status_code=400, detail="Soru boş olamaz. 'question' alanına bir metin gönder.")

    user_content = question

    return call_llm_json(
        model_name=OPENAI_MODEL_OTOBOT,
        system_prompt=SYSTEM_PROMPT_OTOBOT,
        user_content=user_content,
        mode="otobot",
        req=req,
    )
