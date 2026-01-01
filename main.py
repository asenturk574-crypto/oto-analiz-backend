import os
import json
from datetime import datetime
from typing import Any, Dict, Optional, List

from fastapi import FastAPI
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

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL_DEFAULT = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_MODEL_NORMAL = os.getenv("OPENAI_MODEL_NORMAL", OPENAI_MODEL_DEFAULT)
OPENAI_MODEL_PREMIUM = os.getenv("OPENAI_MODEL_PREMIUM", OPENAI_MODEL_DEFAULT)
OPENAI_MODEL_COMPARE = os.getenv("OPENAI_MODEL_COMPARE", OPENAI_MODEL_DEFAULT)
OPENAI_MODEL_OTOBOT = os.getenv("OPENAI_MODEL_OTOBOT", OPENAI_MODEL_DEFAULT)

client = None
if OPENAI_API_KEY and OpenAI is not None:
    client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="Oto Analiz API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # prod'da domain bazlı kısıtla
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# MODELS
# =========================================================
class UsageProfile(BaseModel):
    yearly_km: Optional[int] = Field(default=None)
    usage_type: Optional[str] = Field(default=None)  # city/highway/mixed
    fuel_pref: Optional[str] = Field(default=None)
    gearbox_pref: Optional[str] = Field(default=None)
    family: Optional[bool] = Field(default=None)


class ListingInfo(BaseModel):
    title: Optional[str] = None
    year: Optional[int] = None
    brand: Optional[str] = None
    model: Optional[str] = None
    trim: Optional[str] = None
    km: Optional[int] = None
    fuel: Optional[str] = None
    gearbox: Optional[str] = None
    price_try: Optional[int] = None
    city: Optional[str] = None
    description: Optional[str] = None


class NormalAnalyzeRequest(BaseModel):
    listing: ListingInfo


class PremiumAnalyzeRequest(BaseModel):
    listing: ListingInfo
    profile: Optional[UsageProfile] = None


class CompareRequest(BaseModel):
    a: ListingInfo
    b: ListingInfo
    profile: Optional[UsageProfile] = None


class ManualCarRequest(BaseModel):
    # manuel girilen araç için kısa form
    brand: str
    model: str
    year: int
    km: int
    fuel: Optional[str] = None
    gearbox: Optional[str] = None
    notes: Optional[str] = None
    profile: Optional[UsageProfile] = None


class OtoBotRequest(BaseModel):
    message: str
    context: Optional[Dict[str, Any]] = None


# =========================================================
# HELPERS (PURE)
# =========================================================
def clamp(x: float, a: float, b: float) -> float:
    return max(a, min(b, x))


def infer_car_name(listing: ListingInfo) -> str:
    parts = []
    if listing.year:
        parts.append(str(listing.year))
    if listing.brand:
        parts.append(listing.brand)
    if listing.model:
        parts.append(listing.model)
    if listing.trim:
        parts.append(listing.trim)
    return " ".join(parts) if parts else (listing.title or "Araç")


def infer_segment(listing: ListingInfo) -> str:
    premium_brands = {"audi", "bmw", "mercedes", "mini", "land rover", "volvo", "lexus", "porsche", "jaguar"}
    b = (listing.brand or "").strip().lower()
    if b in premium_brands:
        return "Premium/üst sınıf"
    t = (listing.title or "").lower()
    if any(k in t for k in ["q2", "q3", "q5", "x1", "x3", "gl", "tiguan", "kuga", "sportage", "tucson", "qashqai"]):
        return "Orta-üst (SUV/Crossover)"
    return "Orta sınıf"


def calc_age(year: Optional[int]) -> Optional[int]:
    if not year:
        return None
    now_year = datetime.now().year
    return max(0, now_year - year)


def missing_fields_uncertainty(listing: ListingInfo) -> int:
    keys = ["year", "brand", "model", "km", "fuel", "gearbox", "price_try"]
    missing = 0
    for k in keys:
        v = getattr(listing, k)
        if v in (None, "", 0):
            missing += 1
    return int(clamp(20 + missing * 10, 20, 85))


def uncertainty_label(score: int) -> str:
    if score <= 25:
        return "düşük"
    if score <= 45:
        return "orta"
    return "yüksek"


def band_try(base: float, spread: float) -> Dict[str, int]:
    lo = int(max(0, round(base * (1 - spread))))
    hi = int(max(lo + 1, round(base * (1 + spread))))
    return {"min": lo, "max": hi}


def calc_yearly_costs(listing: ListingInfo, profile: Optional[UsageProfile], segment: str) -> Dict[str, Any]:
    km_year = profile.yearly_km if profile and profile.yearly_km else 15000
    km_total = listing.km or 120000
    age = calc_age(listing.year) or 7

    seg_mult = 1.0
    if "Premium" in segment:
        seg_mult = 1.35
    elif "Orta-üst" in segment:
        seg_mult = 1.15

    wear = (age / 10.0) + (km_total / 200000.0)
    wear = clamp(wear, 0.6, 2.0)

    base_maint = (18000 + km_year * 0.65) * seg_mult * wear
    maint = band_try(base_maint, spread=clamp(0.35 + (wear - 1) * 0.15, 0.35, 0.65))

    fuel = (listing.fuel or (profile.fuel_pref if profile else "") or "").lower()
    if "diesel" in fuel or "dizel" in fuel:
        per_km = 2.6
    elif "hybrid" in fuel or "hibrit" in fuel:
        per_km = 2.0
    elif "ev" in fuel or "electric" in fuel:
        per_km = 1.2
    else:
        per_km = 3.1  # benzin
    base_fuel = km_year * per_km * seg_mult
    fuel_band = band_try(base_fuel, spread=0.35)

    mtv = 5220 if "Premium" not in segment else 8000
    muayene = 1644

    base_ins = 12000 * seg_mult * (1.0 + age * 0.03)
    traffic = band_try(base_ins, 0.35)
    casco = band_try(base_ins * 5.5, 0.45)

    total_avg = int(round(
        (maint["min"] + maint["max"]) / 2 +
        (fuel_band["min"] + fuel_band["max"]) / 2 +
        mtv + muayene +
        (traffic["min"] + traffic["max"]) / 2 +
        (casco["min"] + casco["max"]) / 2
    ))

    return {
        "yearly_km": km_year,
        "maintenance_try": maint,
        "fuel_energy_try": fuel_band,
        "mtv_try": {"min": mtv, "max": mtv},
        "inspection_try": {"min": muayene, "max": muayene},
        "traffic_insurance_try": traffic,
        "casco_try": casco,
        "total_avg_try": total_avg,
    }


def weighted_avg(pairs: List[tuple]) -> int:
    s = 0.0
    w = 0.0
    for val, weight in pairs:
        s += float(val) * float(weight)
        w += float(weight)
    return int(round(s / w)) if w else 0


def pick_base_scores(listing: ListingInfo, profile: Optional[UsageProfile], segment: str) -> Dict[str, int]:
    km_total = listing.km or 120000
    age = calc_age(listing.year) or 7

    wear = clamp((age / 10.0) + (km_total / 200000.0), 0.6, 2.0)

    mech = 82 - (wear - 0.8) * 18
    elec = 80 - (age * 2.2) - (max(0, km_total - 150000) / 50000) * 4

    km_year = profile.yearly_km if profile and profile.yearly_km else 15000
    fuel = (listing.fuel or (profile.fuel_pref if profile else "") or "").lower()
    econ = 70
    if km_year >= 25000:
        econ += 6
    if "diesel" in fuel or "dizel" in fuel:
        econ += 4 if km_year >= 20000 else 1
    if "Premium" in segment:
        econ -= 6

    comfort = 72
    if "Premium" in segment:
        comfort += 6
    elif "Orta-üst" in segment:
        comfort += 3

    body = 78 - (age * 1.2) - (max(0, km_total - 160000) / 60000) * 3

    family = 66
    if profile and profile.family is True:
        if "SUV" in segment or "Orta-üst" in segment:
            family += 6
        else:
            family -= 3

    second_hand = 62
    if "Premium" in segment:
        second_hand = 58
    if km_total >= 170000:
        second_hand -= 5

    fit = 72
    if profile:
        if (profile.usage_type or "").lower() == "mixed":
            fit += 2
        if profile.yearly_km and profile.yearly_km >= 30000:
            fit += 4
        if profile.fuel_pref and ("diesel" in profile.fuel_pref.lower() or "dizel" in profile.fuel_pref.lower()):
            fit += 2
        if "Premium" in segment and profile.yearly_km and profile.yearly_km < 12000:
            fit -= 2

    def c(x: float) -> int:
        return int(clamp(round(x), 0, 100))

    return {
        "mechanic": c(mech),
        "electronics": c(elec),
        "economy": c(econ),
        "comfort": c(comfort),
        "body": c(body),
        "family": c(family),
        "second_hand": c(second_hand),
        "personal_fit": c(fit),
    }


def calc_overall(scores: Dict[str, int]) -> int:
    return weighted_avg([
        (scores["mechanic"], 0.22),
        (scores["electronics"], 0.18),
        (scores["economy"], 0.16),
        (scores["body"], 0.10),
        (scores["comfort"], 0.08),
        (scores["family"], 0.08),
        (scores["second_hand"], 0.08),
        (scores["personal_fit"], 0.10),
    ])


def build_flags(listing: ListingInfo, profile: Optional[UsageProfile], segment: str) -> List[Dict[str, str]]:
    flags: List[Dict[str, str]] = []
    km_total = listing.km or 0
    age = calc_age(listing.year) or 0
    km_year = profile.yearly_km if profile and profile.yearly_km else None
    fuel = (listing.fuel or (profile.fuel_pref if profile else "") or "").lower()

    if km_year and km_year >= 25000 and ("diesel" in fuel or "dizel" in fuel):
        flags.append({"level": "green", "text": "Yıllık km yüksek + dizel tercih: kullanım mantığı daha uyumlu."})

    if km_total >= 160000:
        flags.append({"level": "yellow", "text": "Km yüksek: bakım disiplini ve kayıt/fatura netliği kritik hale gelir."})
    if "Premium" in segment:
        flags.append({"level": "yellow", "text": "Premium segment: parça/işçilik ve sigorta kalemleri bütçede daha hissedilir."})

    if age >= 10:
        flags.append({"level": "red", "text": "Araç yaşı 10+ ise elektronik/konfor aksamında yaşa bağlı risk artar (tahmini)."})
    return flags


def part_service_indices(listing: ListingInfo, segment: str) -> Dict[str, Any]:
    brand = (listing.brand or "").lower()

    availability = 3
    cost_idx = 3
    service_net = 3
    liquidity = 3

    if "Premium" in segment:
        availability = 2
        cost_idx = 4
        service_net = 3
        liquidity = 3

    if brand in ["toyota", "renault", "fiat", "hyundai"]:
        availability += 1
        cost_idx -= 1

    availability = int(clamp(availability, 1, 5))
    cost_idx = int(clamp(cost_idx, 1, 5))
    service_net = int(clamp(service_net, 1, 5))
    liquidity = int(clamp(liquidity, 1, 5))

    def explain(name: str, score: int) -> Dict[str, str]:
        if name == "availability":
            if score <= 2:
                return {
                    "why": "Segment/marka nedeniyle bazı kritik parçalarda stok/termin riski daha yüksek olabilir.",
                    "action": "Kritik parçalar (far/airbag/elektronik modül) için servisle tedarik süresi ve muadil seçenekleri konuş."
                }
            return {
                "why": "Parça bulunurluğu genelde yeterli; yine de kritik parçalarda termin değişebilir.",
                "action": "Ekspertizde çıkan parçalar için 2-3 yerden fiyat ve tedarik süresi sor."
            }
        if name == "cost":
            if score >= 4:
                return {
                    "why": "Premium/üst sınıfta OEM parça + işçilik maliyeti yükselmeye eğilimlidir.",
                    "action": "Bakım kalemlerinde OEM/muadil farkını öğren; planlı bakım bütçesi çıkar."
                }
            return {
                "why": "Parça/işçilik maliyetleri genel ortalamaya yakın.",
                "action": "Yılda 1 büyük + 1 küçük bakım senaryosu için net fiyat al."
            }
        if name == "service":
            if score <= 2:
                return {
                    "why": "Yetkin servis/usta bulma bazı şehirlerde zor olabilir.",
                    "action": "Bulunduğun ilde 2-3 uzman servis seçip ekspertiz/OBD fiyatı al."
                }
            return {
                "why": "Servis erişimi ortalama; şehir/bölgeye göre değişir.",
                "action": "En az 2 alternatif servis listesi tut."
            }
        if score <= 2:
            return {
                "why": "Talep dar olabilir; doğru fiyatlama ve temiz geçmiş daha kritik.",
                "action": "Aynı segmentte 3-5 emsal ilanla (km/hasar/paket) kıyaslayıp fiyat mantığını kontrol et."
            }
        return {
            "why": "Likidite ortalama; temiz geçmiş + doğru paket hızlı satışı destekler.",
            "action": "Doğru fiyatlama için emsal karşılaştırması yap."
        }

    return {
        "part_availability": {"score_5": availability, **explain("availability", availability)},
        "part_cost": {"score_5": cost_idx, **explain("cost", cost_idx)},
        "service_network": {"score_5": service_net, **explain("service", service_net)},
        "second_hand_liquidity": {"score_5": liquidity, **explain("liquidity", liquidity)},
    }


def build_premium_report(listing: ListingInfo, profile: Optional[UsageProfile]) -> Dict[str, Any]:
    car_name = infer_car_name(listing)
    segment = infer_segment(listing)
    age = calc_age(listing.year)
    km_total = listing.km

    uncertainty_score = missing_fields_uncertainty(listing)
    costs = calc_yearly_costs(listing, profile, segment)
    scores_sub = pick_base_scores(listing, profile, segment)
    overall = calc_overall(scores_sub)

    flags = build_flags(listing, profile, segment)
    indices = part_service_indices(listing, segment)

    risks: List[str] = []
    if "Premium" in segment:
        risks.append("Premium sınıfta elektronik/konfor parçaları ve işçilik maliyetleri daha yüksek eğilimlidir (tahmini).")
    if age is not None and age >= 8:
        risks.append("Yaş yükseldikçe sensörler, ekran/aydınlatma, konfor modüllerinde yaşa bağlı arıza olasılığı artar (tahmini).")
    fuel = (listing.fuel or (profile.fuel_pref if profile else "") or "").lower()
    if "diesel" in fuel or "dizel" in fuel:
        risks.append("Dizelde şehir içi kısa mesafe ağırlığında DPF/EGR/enjektör sağlığı daha kritik olabilir; kullanım tipine göre değişir.")
    if km_total is not None and km_total >= 160000:
        risks.append("Km yüksek olduğu için bakım kaydı/fatura ve doğru yağ/bakım periyodu netliği skoru çok etkiler.")

    one_glance = [
        f"Skor: {overall}/100 | Segment: {segment}",
        f"Belirsizlik: {uncertainty_label(uncertainty_score)} (puan: {uncertainty_score}/100)",
        f"Yıllık bakım bandı: {costs['maintenance_try']['min']:,} – {costs['maintenance_try']['max']:,} TL".replace(",", "."),
        f"Yıllık yakıt/enerji bandı: {costs['fuel_energy_try']['min']:,} – {costs['fuel_energy_try']['max']:,} TL".replace(",", "."),
        "Netleştirme: Tramer + servis kayıtları + OBD taraması raporu en çok netleştirir.",
    ]

    value_bullets = [
        "Bu bölüm kesin hüküm içermez; sadece pazarlık yaklaşımı ve belirsizlik yönetimi içindir.",
        "Ekspertizde çıkan masrafları (lastik, fren, alt takım, bakım) kalem kalem fiyatlandırıp pazarlık maddesi yap.",
        "Bakım kayıtları/faturalar yoksa yakın vade bakım bütçesini (yağ/filtre/sıvılar) argümanlaştır.",
        "Tramer + parça değişimi bilgilerini (şasi/podye kontrolüyle birlikte) teyit edip belirsizliği pazarlıkta kullan."
    ]

    clarify_plan = [
        f"Belirsizlik seviyesi: {uncertainty_label(uncertainty_score)} (puan: {uncertainty_score}/100).",
        "Netleştirmek için en çok etki eden 3 adım: Tramer + servis kayıtları + OBD taraması.",
        "Test sürüşünde vites geçişleri, fren ve süspansiyon seslerini özellikle dinle.",
    ]

    checklist = [
        "Rutin bakım kalemlerinin (yağ/filtre/fren) tarihini belgeyle doğrula.",
        "Lastik diş derinliği + balans + fren performansını test sürüşünde kontrol et.",
        "Dizelse: DPF/EGR durumu, enjektör kaçak testi ve turbo basıncını kontrol ettir.",
        "Elektronik modüller (ekran, far, sensörler, cam tavan) sorunsuz çalışmalı.",
        "Şanzıman: soğuk/ısınmış test, adaptasyon ve kaçak kontrolü yaptır.",
        "Turbo/yağ tüketimi, kompresyon ve OBD hata taraması yapılması faydalıdır.",
    ]

    final_summary = [
        "Bu rapor; yaş/km, segment ve profil bilgilerine göre tahmini bant üretir; kesin tespit değildir.",
        "Kesin netlik için: ekspertiz + tramer + OBD birlikte değerlendirilmelidir.",
        "Veri arttıkça belirsizlik düşer ve maliyet/riske dair bantlar daralır."
    ]
    if listing.year and listing.km:
        final_summary.insert(0, f"{listing.year} model, {listing.km:,} km bilgisiyle genel tablo çıkarıldı.".replace(",", "."))

    report = {
        "car_name": car_name,
        "meta": {
            "segment": segment,
            "age": age,
            "km": km_total,
            "fuel": listing.fuel,
            "gearbox": listing.gearbox,
            "yearly_km": profile.yearly_km if profile else None,
            "usage_type": profile.usage_type if profile else None,
        },
        "uncertainty": {"score": uncertainty_score, "label": uncertainty_label(uncertainty_score)},
        "one_glance": {"bullets": one_glance},
        "scores": {
            "overall": overall,
            "sub": {
                "Mekanik": scores_sub["mechanic"],
                "Kaporta": scores_sub["body"],
                "Elektronik/Donanım": scores_sub["electronics"],
                "Ekonomi": scores_sub["economy"],
                "Konfor": scores_sub["comfort"],
                "Aile kullanımı": scores_sub["family"],
                "2. el": scores_sub["second_hand"],
                "Kişiye uygunluk": scores_sub["personal_fit"],
            },
        },
        "flags": flags,
        "costs": costs,
        "risks": {"bullets": risks if risks else ["Belirgin ek risk paterni görünmüyor; yine de ekspertiz/OBD önerilir."]},
        "indices": indices,
        "value_and_bargain": {"bullets": value_bullets},
        "clarify_plan": {"bullets": clarify_plan},
        "checklist": {"bullets": checklist},
        "final_summary": {"bullets": final_summary},
    }
    return report


def llm_rewrite_bullets(report: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """
    Sayıları değiştirmeden metinleri daha doğal yapar.
    Client yoksa dokunmaz.
    """
    if client is None:
        return report

    try:
        payload = {"task": "Rewrite Turkish bullet texts to be short, natural and non-generic. Do NOT change any numbers. Output JSON: {report: <updated>}.", "report": report}
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Concise Turkish. Never change numeric values. Output valid JSON."},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
            temperature=0.35,
        )
        content = resp.choices[0].message.content or "{}"
        obj = json.loads(content)
        new_report = obj.get("report")
        if isinstance(new_report, dict):
            # numeric integrity
            new_report["scores"] = report["scores"]
            new_report["costs"] = report["costs"]
            new_report["uncertainty"] = report["uncertainty"]
            new_report["meta"] = report["meta"]
            new_report["car_name"] = report["car_name"]
            return new_report
    except Exception:
        pass

    return report


# =========================================================
# ROUTES
# =========================================================
@app.get("/health")
def health():
    return {"ok": True, "ts": datetime.utcnow().isoformat()}


@app.post("/normal/analyze")
def normal_analyze(req: NormalAnalyzeRequest):
    """
    Normal: Premium kadar derin değil, daha kısa.
    """
    listing = req.listing
    car_name = infer_car_name(listing)
    segment = infer_segment(listing)
    uncertainty = missing_fields_uncertainty(listing)

    # normal skor = premium hesaplarının sade hali
    scores_sub = pick_base_scores(listing, None, segment)
    overall = calc_overall(scores_sub)

    report = {
        "car_name": car_name,
        "segment": segment,
        "uncertainty": {"score": uncertainty, "label": uncertainty_label(uncertainty)},
        "summary": {
            "bullets": [
                f"Genel skor: {overall}/100",
                f"Belirsizlik: {uncertainty_label(uncertainty)} (puan: {uncertainty}/100)",
                "Kısa öneri: Tramer + ekspertiz + test sürüşü ile karar netleşir.",
            ]
        },
        "scores": {
            "overall": overall,
            "sub": {
                "Mekanik": scores_sub["mechanic"],
                "Elektronik": scores_sub["electronics"],
                "Ekonomi": scores_sub["economy"],
                "2. el": scores_sub["second_hand"],
            },
        },
        "notes": {
            "bullets": [
                "Bu değerlendirme ilan bilgisinin seviyesine göre değişir; eksik bilgi belirsizliği artırır.",
                "Kesin tespit için ekspertiz/OBD şarttır."
            ]
        }
    }

    report = llm_rewrite_bullets(report, OPENAI_MODEL_NORMAL)
    return {"ok": True, "report": report}


@app.post("/premium/analyze")
def premium_analyze(req: PremiumAnalyzeRequest):
    """
    Premium: JSON report döner -> Flutter kart UI
    """
    base_report = build_premium_report(req.listing, req.profile)
    final_report = llm_rewrite_bullets(base_report, OPENAI_MODEL_PREMIUM)
    return {"ok": True, "report": final_report}


@app.post("/compare/analyze")
def compare_analyze(req: CompareRequest):
    """
    Compare: iki aracı aynı şablonda değerlendirir + kısa karşılaştırma özeti
    """
    ra = build_premium_report(req.a, req.profile)
    rb = build_premium_report(req.b, req.profile)

    a_score = ra["scores"]["overall"]
    b_score = rb["scores"]["overall"]

    winner = "A" if a_score > b_score else "B" if b_score > a_score else "Eşit"

    report = {
        "a": {"car_name": ra["car_name"], "overall": a_score, "key": ra["one_glance"]["bullets"][:3]},
        "b": {"car_name": rb["car_name"], "overall": b_score, "key": rb["one_glance"]["bullets"][:3]},
        "result": {
            "winner": winner,
            "bullets": [
                f"A genel skor: {a_score}/100, B genel skor: {b_score}/100",
                "Skor farkı küçükse karar; bakım geçmişi, tramer ve ekspertiz sonucu ile netleşir.",
                "Yıllık maliyet bandı ve belirsizlik düzeyi kıyaslanmalıdır."
            ]
        }
    }

    report = llm_rewrite_bullets(report, OPENAI_MODEL_COMPARE)
    return {"ok": True, "report": report}


@app.post("/manual/analyze")
def manual_analyze(req: ManualCarRequest):
    """
    Manuel: listing gibi davranır, premium rapor üretir
    """
    listing = ListingInfo(
        brand=req.brand,
        model=req.model,
        year=req.year,
        km=req.km,
        fuel=req.fuel,
        gearbox=req.gearbox,
        description=req.notes,
        title=f"{req.year} {req.brand} {req.model}"
    )
    base_report = build_premium_report(listing, req.profile)
    final_report = llm_rewrite_bullets(base_report, OPENAI_MODEL_PREMIUM)
    return {"ok": True, "report": final_report}


@app.post("/otobot/chat")
def otobot_chat(req: OtoBotRequest):
    """
    OtoBot: genel chat. İstersen context içine report/listing vs basarsın.
    """
    if client is None:
        # LLM yoksa basit fallback
        return {
            "ok": True,
            "answer": "Şu an sohbet modu (OtoBot) için model anahtarı yok. Premium/normal analizler çalışır. OtoBot'u açmak için OPENAI_API_KEY gerekli."
        }

    context = req.context or {}
    payload = {
        "role": "OtoBot",
        "rule": "Kısa, net ve güvenli konuş. Kesin hüküm verme; öneri dilinde kal.",
        "context": context,
        "message": req.message
    }

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL_OTOBOT,
            messages=[
                {"role": "system", "content": "You are OtoBot. Turkish, concise. Avoid definitive claims. Provide actionable steps."},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.5,
        )
        answer = (resp.choices[0].message.content or "").strip()
        return {"ok": True, "answer": answer}
    except Exception as e:
        return {"ok": False, "error": str(e)}
