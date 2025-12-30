import os
import json
from typing import Any, Dict, List, Optional

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

    # Flutter tarafında gönderdiğin birleşik metin (Araç:..., beklentiler, ilan açıklaması vb.)
    text: Optional[str] = None

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
# Backend tahminleri (segment + maliyet)
# ---------------------------------------------------------
def guess_segment(vehicle: Vehicle) -> str:
    name = (vehicle.make + " " + vehicle.model).lower()

    b_hatch = ["clio", "fiesta", "yaris", "i20", "polo", "corsa", "fabia"]
    c_sedan = ["corolla", "focus", "megane", "civic", "astra", "egea"]
    c_suv = ["duster", "kuga", "qashqai", "sportage", "tucson", "3008"]
    d_sedan = ["passat", "superb", "508", "insignia"]
    premium = ["bmw", "mercedes", "audi", "volvo", "range rover"]

    if any(k in name for k in b_hatch):
        return "B-segment küçük hatchback"
    if any(k in name for k in c_sedan):
        return "C-segment aile sedan/hatchback"
    if any(k in name for k in c_suv):
        return "C-segment SUV"
    if any(k in name for k in d_sedan):
        return "D-segment konfor sedan"
    if any(k in name for k in premium):
        return "premium D/E-segment"

    return "genel C/D-segment binek araç"


def estimate_costs_and_risks(req: AnalyzeRequest) -> Dict[str, Any]:
    v = req.vehicle
    p = req.profile or Profile()

    segment = guess_segment(v)
    age: Optional[int] = None
    if v.year:
        age = max(0, 2025 - v.year)

    mileage = v.mileage_km or 0

    base_maintenance = 15000  # TL
    base_fuel = 25000

    seg_mult = 1.0
    if "B-segment" in segment:
        seg_mult = 0.7
    elif "C-segment" in segment:
        seg_mult = 1.0
    elif "SUV" in segment:
        seg_mult = 1.3
    elif "premium" in segment or "D-segment" in segment:
        seg_mult = 1.6

    age_mult = 1.0
    if age is not None:
        if age > 12:
            age_mult = 1.6
        elif age > 8:
            age_mult = 1.3
        elif age > 5:
            age_mult = 1.1

    km_mult = 1.0
    if mileage > 250_000:
        km_mult = 1.7
    elif mileage > 180_000:
        km_mult = 1.4
    elif mileage > 120_000:
        km_mult = 1.2

    fuel_mult = 1.0
    fuel_risk = "orta"
    if v.fuel == "diesel":
        fuel_mult = 1.1
        if mileage > 180_000 and p.usage == "city":
            fuel_risk = "yüksek (DPF / enjektör riski)"
    elif v.fuel == "lpg":
        fuel_mult = 0.9
        fuel_risk = "orta (LPG montaj kalitesine bağlı)"
    elif v.fuel in ("hybrid", "electric"):
        fuel_mult = 0.8
        fuel_risk = "düşük-orta (batarya sağlığına bağlı)"

    yearly_maintenance = int(base_maintenance * seg_mult * age_mult * km_mult)
    yearly_fuel = int(base_fuel * seg_mult * ((p.yearly_km / 15000) or 1) * fuel_mult)

    if "premium" in segment:
        insurance_level = "yüksek"
    elif "SUV" in segment or "D-segment" in segment:
        insurance_level = "orta-yüksek"
    else:
        insurance_level = "orta"

    if "C-segment" in segment or "B-segment" in segment:
        resale_speed = "hızlı"
    elif "SUV" in segment:
        resale_speed = "orta-hızlı"
    else:
        resale_speed = "orta"

    risk_level = "orta"
    risk_notes: List[str] = []
    if age is not None and age > 12:
        risk_level = "yüksek"
        risk_notes.append("İleri yaş nedeniyle kronik masraflar artabilir.")
    if mileage > 250_000:
        risk_level = "yüksek"
        risk_notes.append("Km çok yüksek, motor/şanzıman revizyon riski.")
    if "yüksek" in fuel_risk:
        risk_level = "yüksek"

    return {
        "segment_guess": segment,
        "age": age,
        "mileage_km": mileage,
        "estimated_yearly_maintenance_tr": yearly_maintenance,
        "estimated_yearly_fuel_tr": yearly_fuel,
        "insurance_level": insurance_level,
        "resale_speed": resale_speed,
        "fuel_risk_comment": fuel_risk,
        "overall_risk_level": risk_level,
        "risk_notes": risk_notes,
    }


# ---------------------------------------------------------
# Kullanıcı prompt'u (LLM için)
# ---------------------------------------------------------
def build_user_content(req: AnalyzeRequest, mode: str) -> str:
    v = req.vehicle or Vehicle()
    p = req.profile or Profile()

    ad_text = (req.ad_description or "").strip()
    free_text = ""
    try:
        free_text = (req.text or "").strip()
    except AttributeError:
        free_text = ""

    all_ss: List[str] = []
    if req.screenshot_base64:
        all_ss.append(req.screenshot_base64)
    if req.screenshots_base64:
        all_ss.extend([s for s in req.screenshots_base64 if s])

    backend_context = estimate_costs_and_risks(req)

    ss_info = ""
    if all_ss:
        ss_info = (
            f"\nKullanıcı {len(all_ss)} adet ilan ekran görüntüsü ekledi. "
            "Bu görüntülerdeki fiyat, donanım, paket ve hasar bilgilerini de analizinde kullan. "
            "Eğer görüntülere doğrudan erişemiyorsan bile, bu bilgilerin mevcut olduğunu "
            "varsayarak genel bir değerlendirme yap."
        )

    if not (v.make.strip() or v.model.strip() or ad_text or free_text or all_ss):
        ad_text = (
            "Kullanıcı çok az bilgi verdi. Türkiye ikinci el piyasasında genel kabul gören "
            "kriterlerle, varsayımsal bir aile aracı analizi yap."
        )

    base_text = f"""
Kullanıcı Oto Analiz uygulamasında **{mode}** modunda analiz istiyor.

Araç bilgileri (boş olan alanlar '-' olabilir):
- Marka: {v.make or "-"}
- Model: {v.model or "-"}
- Yıl: {v.year or "-"}
- Kilometre: {v.mileage_km or "-"} km
- Yakıt: {v.fuel or p.fuel_preference}

Kullanım profili (tahmini değerler olabilir):
- Yıllık km: {p.yearly_km} km
- Kullanım tipi: {p.usage}
- Yakıt tercihi: {p.fuel_preference}
""".strip()

    if ad_text:
        base_text += f"\n\nİlan açıklaması veya satıcının notu:\n{ad_text}"

    if free_text and free_text != ad_text:
        base_text += (
            "\n\nKullanıcının kendi yazdığı ek bilgiler (beklentiler, bütçe, öğrenci/ilk araç vb.):\n"
            f"{free_text}"
        )

    base_text += "\n\n--- Backend tahmini maliyet & risk bilgileri (kaba hesap) ---\n"
    base_text += json.dumps(backend_context, ensure_ascii=False)
    base_text += "\n-----------------------------------------------------------\n"
    base_text += ss_info

    return base_text.strip()


# ---------------------------------------------------------
# Fallback JSON üreticileri (LLM patlarsa)
# ---------------------------------------------------------
def fallback_normal(req: AnalyzeRequest) -> Dict[str, Any]:
    v = req.vehicle
    seg_info = estimate_costs_and_risks(req)
    title = f"{v.year or ''} {v.make} {v.model}".strip() or "Araç Analizi"

    return {
        "scores": {
            "overall_100": 70,
            "mechanical_100": 70,
            "body_100": 70,
            "economy_100": 70,
        },
        "summary": {
            "short_comment": "Araç hakkında temel bilgilere göre genel bir değerlendirme yapıldı.",
            "pros": [
                "Türkiye piyasasına göre makul bir ikinci el tercih olabilir.",
                "Doğru bakım ve ekspertiz ile uzun süre kullanılabilir.",
            ],
            "cons": [
                "Ekspertiz ve tramer yapılmadan net yoruma gidilemez.",
                "Bakım geçmişi ve km durumu mutlaka detaylı kontrol edilmelidir.",
            ],
            "estimated_risk_level": seg_info.get("overall_risk_level", "orta"),
        },
        "preview": {
            "title": title,
            "price_tag": "Normal",
            "spoiler": "Sınırlı bilgiye göre genel, nötr bir ikinci el değerlendirmesi yapıldı. Detaylı ekspertiz mutlaka önerilir.",
            "bullets": [
                "Ekspertiz ve tramer kaydı mutlaka kontrol edilmeli.",
                "Bakım kayıtları ve km uyumu doğrulanmalı.",
                "Lastik ve fren durumu pazarlıkta avantaj sağlayabilir.",
            ],
        },
    }


def fallback_premium(req: AnalyzeRequest) -> Dict[str, Any]:
    v = req.vehicle
    seg_info = estimate_costs_and_risks(req)
    title = f"{v.year or ''} {v.make} {v.model}".strip() or "Araç Analizi (Premium)"

    yearly_maint = seg_info.get("estimated_yearly_maintenance_tr", 15000)
    yearly_fuel = seg_info.get("estimated_yearly_fuel_tr", 25000)
    segment = seg_info.get("segment_guess", "C/D-segment binek araç")
    risk_level = seg_info.get("overall_risk_level", "orta")

    p = req.profile or Profile()

    paragraphs: List[str] = []

    paragraphs.append(
        f"{v.year or 'Bilinmeyen yıl'} model {v.make or ''} {v.model or ''} için, "
        f"verdiğin sınırlı bilgilere göre premium formatta bir değerlendirme yaptım. "
        f"Araç genel olarak {segment} sınıfında değerlendirilebilir ve Türkiye ikinci el "
        f"piyasasında benzer araçlarla benzer beklentiler taşır."
    )

    paragraphs.append(
        f"Kullanım profilin yıllık yaklaşık {p.yearly_km} km ve '{p.usage}' kullanım tipi şeklinde. "
        f"Bu durumda aracın yıllık bakım maliyetinin kabaca {yearly_maint} TL, yakıt maliyetinin ise "
        f"yaklaşık {yearly_fuel} TL civarında seyredeceğini varsayabiliriz. Bu rakamlar, yaş, kilometre "
        f"ve segmenti dikkate alan kaba tahminlerdir; gerçek değerler aracın durumuna, bakım geçmişine "
        f"ve yakıt fiyatlarına göre değişebilir."
    )

    paragraphs.append(
        "Satın alma öncesinde detaylı ekspertiz, tramer ve bakım kaydı kontrolü yapmak çok önemli. "
        "Özellikle motor, şanzıman ve yürüyen aksamda pahalı çıkabilecek kronik sorunlar açısından "
        "yetkili veya güvenilir bir servisten destek almakta fayda var. Eğer öğrenciysen ya da ilk "
        "aracını alıyorsan, bütçeni zorlamayan ama masraf çıkarsa karşılayabileceğin bir seviyede "
        "kalmaya dikkat etmelisin."
    )

    result_text = "\n\n".join(paragraphs)

    return {
        "scores": {
            "overall_100": 75,
            "mechanical_100": 74,
            "body_100": 73,
            "economy_100": 70,
            "comfort_100": 72,
            "family_use_100": 78,
            "resale_100": 76,
        },
        "cost_estimates": {
            "yearly_maintenance_tr": yearly_maint,
            "yearly_fuel_tr": yearly_fuel,
            "insurance_level": seg_info.get("insurance_level", "orta"),
            "notes": "Hesaplamalar sınırlı bilgiye göre tahmini olarak yapılmıştır; "
                     "gerçek maliyetler araç durumuna göre değişebilir.",
        },
        "risk_analysis": {
            "chronic_issues": [
                "İleri km ve yaşa bağlı motor/şanzıman yıpranma ihtimali.",
                "Süspansiyon ve yürüyen aksamda bakım ihtiyacı çıkabilir.",
            ],
            "risk_level": risk_level,
            "warnings": [
                "Satın almadan önce kapsamlı ekspertiz ve tramer sorgusu yaptırılması önerilir.",
                "Bakım geçmişi ve km uyumu mutlaka teyit edilmelidir.",
            ],
        },
        "summary": {
            "short_comment": (
                "Verilen bilgilere göre genel olarak dengeli, ancak bakım ve km durumuna "
                "bağlı riskleri olan bir ikinci el tercih gibi görünüyor."
            ),
            "pros": [
                "Segmentine göre makul kullanım maliyetleri.",
                "Doğru bakım ile uzun süre sorunsuz kullanılma potansiyeli.",
                "Türkiye piyasasında parçaya erişim genellikle kolay.",
            ],
            "cons": [
                "Yaş ve km yükseldikçe beklenmeyen masraf riski artar.",
                "Bakım geçmişi belirsizse şans faktörü devreye girer.",
            ],
            "who_should_buy": (
                "Bütçesini bilen, ekspertize ve bakıma bütçe ayırmayı kabul eden, aile veya "
                "günlük kullanım odaklı kullanıcılar için uygun olabilir."
            ),
        },
        "result": result_text,
        "preview": {
            "title": title,
            "price_tag": "Normal",
            "spoiler": (
                "Sınırlı bilgiyle yapılan, bakım ve maliyetlere özel vurgu yapan premium değerlendirme. "
                "Ekspertiz ve tramer kontrolü mutlaka yapılmalıdır."
            ),
            "bullets": [
                "Tahmini yıllık bakım ve yakıt maliyeti orta seviyede.",
                "İkinci el piyasasında alıcı bulma potansiyeli fena değil.",
                "Satın almadan önce detaylı ekspertiz şart.",
            ],
        },
    }


def fallback_manual(req: AnalyzeRequest) -> Dict[str, Any]:
    # Manuel analiz, normal fallback'e çok benzer
    return fallback_normal(req)


def fallback_compare(req: CompareRequest) -> Dict[str, Any]:
    left_title = (f"{req.left.vehicle.make} {req.left.vehicle.model}").strip() or "Sol araç"
    right_title = (f"{req.right.vehicle.make} {req.right.vehicle.model}").strip() or "Sağ araç"

    return {
        "better_overall": "left",
        "summary": f"{left_title} genel kullanım için biraz daha dengeli bir tercih olarak varsayıldı. Ancak her iki araç için de ekspertiz ve tramer şarttır.",
        "left_pros": [
            f"{left_title} için varsayımsal olarak daha dengeli maliyet/performans oranı kabul edildi.",
            "Aile ve karışık kullanım için uygun olabilir.",
        ],
        "left_cons": [
            "Gerçek durum bilinmediği için mutlaka yerinde inceleme gerekir.",
        ],
        "right_pros": [
            f"{right_title} da doğru bakımla mantıklı bir tercih olabilir.",
        ],
        "right_cons": [
            "Toplanan bilgilere göre maliyet veya kullanım açısından biraz daha dikkatli incelenmelidir.",
        ],
        "use_cases": {
            "family_use": f"Aile kullanımı için {left_title} biraz daha avantajlı varsayılmıştır.",
            "long_distance": "Her iki araç da düzenli bakım ile uzun yolda kullanılabilir.",
            "city_use": "Şehir içi kullanımda yakıt ve konfor açısından her iki aracın da test edilmesi önerilir.",
        },
    }


def fallback_otobot(question: str) -> Dict[str, Any]:
    return {
        "answer": "Verdiğin bilgiler sınırlı olsa da, Türkiye'de genelde C-segment bir dizel veya benzinli-hybrid araç; aile, konfor ve uzun yol dengesi için mantıklı bir başlangıç noktasıdır. Yıllık km yüksekse dizel veya ekonomik benzinli, daha düşükse benzinli veya hybrid düşünülebilir. Satın almadan önce mutlaka ekspertiz, tramer ve bakım geçmişi kontrolü yaptır.",
        "suggested_segments": ["C-sedan", "C-hatchback", "C-SUV"],
        "example_models": [
            "Toyota Corolla",
            "Hyundai i30 / Elantra",
            "Renault Megane",
            "Honda Civic",
        ],
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
    # OpenAI client yoksa direkt fallback
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
            parsed = json.loads(content)
        else:
            parsed = content  # type: ignore[assignment]

        # Premium analizde model "result" alanını boş bırakırsa,
        # summary'den otomatik bir metin derle.
        if mode == "premium" and isinstance(parsed, dict):
            if not parsed.get("result"):
                summary = parsed.get("summary") or {}
                short_comment = summary.get("short_comment") or ""
                pros = summary.get("pros") or []
                cons = summary.get("cons") or []

                parts: List[str] = []
                if short_comment:
                    parts.append(str(short_comment))
                if pros:
                    parts.append(
                        "Artılar:\n- " + "\n- ".join(str(p) for p in pros[:6])
                    )
                if cons:
                    parts.append(
                        "Eksiler:\n- " + "\n- ".join(str(c) for c in cons[:6])
                    )

                if parts:
                    parsed["result"] = "\n\n".join(parts)

        return parsed
    except Exception as e:
        # Her türlü OpenAI / JSON hatasında fallback kullan
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
Sen 'Oto Analiz' uygulaması için çalışan bir ARAÇ İLANI ANALİZ ASİSTANI'sın.

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
  }
}

Kurallar:
- Tüm alanlar JSON içinde mutlaka olsun (boş bile kalsa).
- PREVIEW kısmı Keşfet için kullanılacak:
  - 'alınır', 'alınmaz', 'sakın', 'riskli', 'tehlikeli' gibi kelimeleri KULLANMA.
  - Fiyatla ilgili sadece 'Uygun/Normal/Yüksek' etiketi ver, rakam yazma.
- Dil: Türkçe.
"""

SYSTEM_PROMPT_PREMIUM = """
Sen 'Oto Analiz' uygulamasının PREMIUM analiz asistanısın.

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
  "result": "",
  "preview": {
    "title": "",
    "price_tag": null,
    "spoiler": "",
    "bullets": []
  }
}

Kurallar:
- Tüm alanlar JSON içinde mutlaka olsun (boş bile kalsa).

- "result" alanı:
  - En az 3 paragraf ve yaklaşık 180–250 kelime uzunluğunda olsun.
  - Araç yaşı, kilometre, yakıt tipi, segment, yıllık km, kullanım tipi ve kullanıcının beklentilerini
    (ör. öğrenciyim, ilk aracım, aile aracı istiyorum vb.) mutlaka yorumla.
  - Tahmini yıllık bakım ve yakıt maliyetlerinden bahset, ama abartma;
    backend tarafından gönderilen tahmini değerleri makul aralıkta kullan.
  - 'alınır', 'alınmaz', 'sakın alma', 'çöp', 'pert' gibi sert ifadeleri kullanma;
    rehberlik eden nötr bir dil kullan.

- cost_estimates.yearly_maintenance_tr ve yearly_fuel_tr:
  - Backend tahminindeki değerleri temel al; bunları 2–3 katına çıkartma.
  - Emin değilsen yaklaşık aralık belirtebilirsin (ör. 20_000–30_000 bandı).

- risk_analysis.chronic_issues ve warnings:
  - Araç yaşına, km'sine ve yakıt tipine göre 3–6 madde üret.

- summary.pros ve summary.cons:
  - Her biri 4–8 madde arasında olsun.
  - Yakıt ekonomisi, parça bulunabilirliği, ikinci el piyasası, konfor, performans
    ve sigorta seviyesi gibi konulara değin.

- preview.price_tag:
  - Yalnızca "Uygun", "Normal" veya "Yüksek" (veya null) kullan; rakam yazma.

Dil: Türkçe.
"""

SYSTEM_PROMPT_MANUAL = """
Sen 'Oto Analiz' uygulamasında KULLANICININ KENDİ ARACI için manuel analiz yapan asistansın.
Kullanıcı bazen çok az bilgi verebilir; bu durumda bile genel, bilgilendirici bir analiz üret.

Çıktı formatın NORMAL analizle aynıdır, yani:

{
  "scores": {...},
  "summary": {...},
  "preview": {...}
}

Kurallar:
- PREVIEW kısmı nötr olsun, 'alınır/alınmaz' gibi ifadeler kullanma.
- Bilgiler çok azsa bile 'ekspertiz, tramer, bakım kaydı' gibi genel tavsiyelere odaklan.
- Dil: Türkçe.
"""

SYSTEM_PROMPT_COMPARE = """
Sen 'Oto Analiz' uygulaması için ARAÇ KARŞILAŞTIRMA asistanısın.
Kullanıcıya iki aracı teknik, maliyet ve kullanım açısından karşılaştır.

Çıktın sadece JSON olsun:
{
  "better_overall": "left" | "right",
  "summary": "kısa genel değerlendirme",
  "left_pros": ["sol araç artıları"],
  "left_cons": ["sol araç eksileri"],
  "right_pros": ["sağ araç artıları"],
  "right_cons": ["sağ araç eksileri"],
  "use_cases": {
    "family_use": "hangi araç daha mantıklı ve neden",
    "long_distance": "...",
    "city_use": "..."
  }
}
Dil: Türkçe, sadece JSON.
"""

SYSTEM_PROMPT_OTOBOT = """
Sen 'Oto Analiz' uygulamasının OTOBOT isimli araç alma rehberisin.
Kullanıcı bütçesini, kullanım şeklini ve beklentisini anlatıyor olabilir.
Görevin: Ona Türkiye piyasasına göre mantıklı segment ve model önerileri vermek.

Çıktı sadece JSON olsun:
{
  "answer": "kullanıcıya verilen detaylı tavsiye yanıtı (Türkçe)",
  "suggested_segments": ["B-SUV", "C-sedan", ...],
  "example_models": ["Örnek model 1", "Örnek model 2", "..."]
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
