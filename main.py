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

    if not (v.make.strip() or v.model.strip() or ad_text or all_ss):
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
"""

    if ad_text:
        base_text += f"\nİlan açıklaması veya kullanıcı notu:\n{ad_text}\n"

    base_text += "\n--- Backend tahmini maliyet & risk bilgileri (kaba hesap) ---\n"
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

    result_text = (
        f"{title} için temel bilgilere göre genel bir değerlendirme:\n"
        f"- Segment tahmini: {seg_info.get('segment_guess', 'bilinmiyor')}.\n"
        f"- Tahmini yıllık bakım maliyeti: yaklaşık "
        f"{seg_info.get('estimated_yearly_maintenance_tr', 15000)} TL civarı.\n"
        f"- Tahmini yıllık yakıt maliyeti: yaklaşık "
        f"{seg_info.get('estimated_yearly_fuel_tr', 25000)} TL civarı.\n"
        f"- Genel risk seviyesi: {seg_info.get('overall_risk_level', 'orta')}.\n\n"
        "Satın almadan önce mutlaka ekspertiz, tramer ve bakım geçmişi kontrolü yaptırılmalı; "
        "özellikle km ve yaş ileriyse motor, şanzıman ve yürür aksam detaylı incelenmelidir."
    )

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
                "C/D segmenti bir araç için konfor ve kullanım dengesi potansiyeli var.",
            ],
            "cons": [
                "Ekspertiz ve tramer yapılmadan net yoruma gidilemez.",
                "Bakım geçmişi ve km durumu mutlaka detaylı kontrol edilmelidir.",
                "Lastik, fren ve yürür aksam masraf çıkarma ihtimali vardır.",
            ],
            "estimated_risk_level": seg_info.get("overall_risk_level", "orta"),
        },
        "preview": {
            "title": title,
            "price_tag": "Normal",
            "spoiler": (
                "Sınırlı bilgiye göre genel, nötr bir ikinci el değerlendirmesi yapıldı. "
                "Detaylı ekspertiz mutlaka önerilir."
            ),
            "bullets": [
                "Ekspertiz ve tramer kaydı mutlaka kontrol edilmeli.",
                "Bakım kayıtları ve km uyumu doğrulanmalı.",
                "Lastik ve fren durumu pazarlıkta avantaj sağlayabilir.",
            ],
        },
        "result": result_text,
    }


def fallback_premium(req: AnalyzeRequest) -> Dict[str, Any]:
    v = req.vehicle
    seg_info = estimate_costs_and_risks(req)
    title = f"{v.year or ''} {v.make} {v.model}".strip() or "Araç Analizi (Premium)"

    result_text = (
        f"Özet:\n"
        f"{title} için verilen yıl, km ve yakıt bilgilerine göre araç genel olarak "
        f"{seg_info.get('segment_guess', 'C/D-segment')} sınıfında, günlük kullanım "
        "ve uzun yol dengesini sunabilecek bir ikinci el aday gibi görünüyor.\n\n"
        "Yıllık Maliyet Tahmini:\n"
        f"- Tahmini yıllık bakım maliyeti: yaklaşık "
        f"{seg_info.get('estimated_yearly_maintenance_tr', 18000)} TL.\n"
        f"- Tahmini yıllık yakıt maliyeti: yaklaşık "
        f"{seg_info.get('estimated_yearly_fuel_tr', 28000)} TL.\n"
        f"- Sigorta/kasko seviyesi: {seg_info.get('insurance_level', 'orta')} seviyede; "
        "bütçe planlanırken bu kalem de hesaba katılmalı.\n\n"
        "Riskler & Dikkat Edilmesi Gerekenler:\n"
        f"- Genel risk seviyesi: {seg_info.get('overall_risk_level', 'orta')}.\n"
        "- Km ve yaş arttıkça motor, şanzıman ve yürür aksam masrafları belirginleşebilir.\n"
        "- Tramer ve ekspertiz raporu olmadan sadece ilan metnine güvenmek yeterli değildir.\n\n"
        "Piyasa Yorumu ve Alternatifler:\n"
        "Bu araç, aynı segmentteki muadilleriyle kıyaslandığında mantıklı bir seçenek olabilir; "
        "ancak fiyat, donanım ve hasar geçmişi benzer ilanlarla mutlaka karşılaştırılmalıdır. "
        "Bütçene yakın alternatif modelleri de inceleyerek pazarlık payını arttırman faydalı olur."
    )

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
            "yearly_maintenance_tr": seg_info.get(
                "estimated_yearly_maintenance_tr", 18000
            ),
            "yearly_fuel_tr": seg_info.get("estimated_yearly_fuel_tr", 28000),
            "insurance_level": seg_info.get("insurance_level", "orta"),
            "notes": (
                "Hesaplamalar sınırlı bilgiye göre tahmini olarak yapılmıştır; gerçek maliyetler "
                "araç durumuna ve kullanım şekline göre değişebilir."
            ),
        },
        "risk_analysis": {
            "chronic_issues": [
                "Bu segmentte tipik ikinci el araçlarda yaşa ve km'ye bağlı standart yıpranma görülebilir.",
                "Özellikle yüksek km’de süspansiyon, şanzıman ve ön takım masraf çıkarabilir.",
            ],
            "risk_level": seg_info.get("overall_risk_level", "orta"),
            "warnings": [
                "Satın almadan önce kapsamlı ekspertiz ve tramer sorgusu yaptırılması önerilir.",
                "Bakım geçmişi ve km uyumu mutlaka servis kayıtları ile teyit edilmelidir.",
                "Lastik, fren ve sarf malzeme durumuna göre ek bütçe ayırmak gerekebilir.",
            ],
        },
        "summary": {
            "short_comment": (
                "Verilen bilgilere göre araç, doğru kontrol ve pazarlıkla mantıklı bir ikinci el tercih "
                "olabilir; ancak km, yaş ve bakım geçmişi mutlaka detaylı incelenmelidir."
            ),
            "pros": [
                "Segmentine göre konfor ve donanım seviyesi tatmin edici olabilir.",
                "Doğru bakım yapıldığında uzun süre kullanılma potansiyeli var.",
                "Piyasada benzer ilanlarla kıyaslandığında alıcı bulma potansiyeli yüksek.",
                "Uzun yol ve aile kullanımı için dengeli bir profil sunabilir.",
                "LPG / dizel / hybrid gibi ekonomik seçenekler maliyetleri aşağı çekebilir (varsa).",
            ],
            "cons": [
                "Yüksek km veya düzensiz bakım geçmişi ciddi masraflar doğurabilir.",
                "Hasar geçmişi ve tramer raporu bilinmeden net karar vermek risklidir.",
                "Pahalı parça ve işçilik gerektiren markalarda yıllık maliyet yukarı çıkabilir.",
                "Kasko ve sigorta maliyetleri, ekonomik modellerden daha yüksek olabilir.",
            ],
            "who_should_buy": (
                "Ailesiyle düzenli kullanım planlayan, bütçesini bilen ve satın almadan önce detaylı "
                "ekspertiz yaptırmaya hazır kullanıcılar için uygun olabilir."
            ),
        },
        "preview": {
            "title": title,
            "price_tag": "Normal",
            "spoiler": (
                "Tahmini bakım ve yakıt maliyetleri orta seviyede. Ekspertiz ve tramer sonrasında, "
                "piyasa ile karşılaştırarak değerlendirilmesi gereken bir ikinci el seçenek."
            ),
            "bullets": [
                "Yıllık tahmini bakım ve yakıt maliyeti orta seviyede.",
                "Piyasada benzer ilanlarla mutlaka kıyaslanmalı.",
                "Ekspertiz ve tramer raporu olmadan karar verilmemeli.",
            ],
        },
        "result": result_text,
    }


def fallback_manual(req: AnalyzeRequest) -> Dict[str, Any]:
    v = req.vehicle
    seg_info = estimate_costs_and_risks(req)
    title = f"{v.year or ''} {v.make} {v.model}".strip() or "Kendi Aracın Analizi"

    result_text = (
        f"Kendi aracın olan {title} için temel bilgilere göre genel bir değerlendirme:\n"
        f"- Segment tahmini: {seg_info.get('segment_guess', 'bilinmiyor')}.\n"
        f"- Tahmini yıllık bakım maliyeti: "
        f"{seg_info.get('estimated_yearly_maintenance_tr', 15000)} TL civarı.\n"
        f"- Tahmini yıllık yakıt maliyeti: "
        f"{seg_info.get('estimated_yearly_fuel_tr', 25000)} TL civarı.\n"
        f"- Genel risk seviyesi: {seg_info.get('overall_risk_level', 'orta')}.\n\n"
        "Aracını uzun süre kullanmayı düşünüyorsan; periyodik bakımları aksatmamak, "
        "tramer ve hasar geçmişini takip etmek ve özellikle yürür aksam seslerini ciddiye "
        "almak uzun vadede seni yüksek masraflardan koruyabilir."
    )

    return {
        "scores": {
            "overall_100": 72,
            "mechanical_100": 72,
            "body_100": 71,
            "economy_100": 70,
        },
        "summary": {
            "short_comment": (
                "Verilen sınırlı bilgilere göre aracın genel durumu ortalama seviyede kabul edilmiştir; "
                "bakımlar düzenli yapılırsa uzun süre kullanılabilir."
            ),
            "pros": [
                "Bakımları düzenli yapıldığında uzun süre sorunsuz kullanılabilir.",
                "Segmentine göre konfor ve kullanım dengesini sağlayabilir.",
                "Aracını tanıdığın için geçmişini bilmek ikinci el araçlara göre avantajlıdır.",
            ],
            "cons": [
                "Yüksek km ve yaş ilerledikçe kronik masraflar artabilir.",
                "Bazı büyük bakım kalemleri (triger, debriyaj, yürür aksam) birikerek yük getirebilir.",
                "Bakım ve onarım geciktirilirse daha büyük masraflara yol açabilir.",
            ],
            "estimated_risk_level": seg_info.get("overall_risk_level", "orta"),
        },
        "preview": {
            "title": title,
            "price_tag": None,
            "spoiler": (
                "Kendi aracın için genel bir durum analizi ve bakım/maliyet tavsiyesi sunuluyor. "
                "Düzenli bakım ve takip ile uzun süre keyifle kullanılabilir."
            ),
            "bullets": [
                "Periyodik bakımları zamanında yaptır.",
                "Tramer ve hasar geçmişini takipte tut.",
                "Şüpheli ses ve titreşimleri önemseyip kontrol ettir.",
            ],
        },
        "result": result_text,
    }


def fallback_compare(req: CompareRequest) -> Dict[str, Any]:
    left_title = (f"{req.left.vehicle.make} {req.left.vehicle.model}").strip() or "Sol araç"
    right_title = (f"{req.right.vehicle.make} {req.right.vehicle.model}").strip() or "Sağ araç"

    return {
        "better_overall": "left",
        "summary": (
            f"{left_title}, varsayımsal olarak genel kullanım ve maliyet dengesi açısından "
            f"{right_title} modeline göre biraz daha avantajlı kabul edilmiştir. "
            "Ancak her iki araç için de ekspertiz ve tramer sonuçları görülmeden net karar verilmemelidir."
        ),
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
        "answer": (
            "Verdiğin bilgiler sınırlı olsa da, Türkiye'de genelde C-segment bir dizel veya "
            "benzinli-hybrid araç; aile, konfor ve uzun yol dengesi için mantıklı bir başlangıç "
            "noktasıdır. Yıllık km yüksekse dizel veya ekonomik benzinli, daha düşükse benzinli "
            "veya hybrid düşünülebilir. Satın almadan önce mutlaka ekspertiz, tramer ve bakım "
            "geçmişi kontrolü yaptır."
        ),
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
            return json.loads(content)
        return content  # type: ignore[return-value]
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
  },
  "result": ""
}

Kurallar:
- Tüm alanlar JSON içinde mutlaka olsun (boş bile kalsa).
- "pros" ve "cons" içinde tercihen en az 4'er madde yaz.
- "result" alanında 3–4 paragraflık detaylı bir açıklama yap:
  - Özet
  - Tahmini maliyet ve riskler
  - Kullanım senaryoları (şehir içi, uzun yol, aile kullanımı)
- PREVIEW kısmı Keşfet için kullanılacak:
  - 'alınır', 'alınmaz', 'sakın', 'riskli', 'tehlikeli' gibi kelimeleri KULLANMA.
  - Fiyatla ilgili sadece 'Uygun/Normal/Yüksek' etiketi ver, rakam yazma.
- Türkiye ikinci el piyasasına göre yorum yap, ama emin olmadığın yerde net rakam üretme.
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
  "preview": {
    "title": "",
    "price_tag": null,
    "spoiler": "",
    "bullets": []
  },
  "result": ""
}

Kurallar:
- Tüm alanlar JSON içinde mutlaka olsun (boş bile kalsa).
- "pros" ve "cons" içinde mümkünse 5–7 detaylı madde yaz.
- "cost_estimates" alanındaki değerleri, sistem mesajında verilen backend tahminiyle uyumlu kullan.
- "result" alanında en az 4–6 paragraflık detaylı bir premium analiz yaz:
  - Özet
  - Yıllık tahmini bakım ve yakıt maliyetleri (aralık olarak bahset, kesin rakam verme)
  - Kronik sorunlar ve olası masraf riskleri
  - Piyasa yorumu (sıfır km fiyatına göre genel olarak pahalı / normal / uygun gibi nitel kıyas)
  - Hangi tip kullanıcı için mantıklı olur
- Kesin fiyat rakamları uydurma; fiyatı sadece nitelik olarak değerlendir (Uygun/Normal/Yüksek, sıfırına yakın, üstünde, altında gibi).
- PREVIEW kısmı Keşfet için kullanılacak:
  - 'alınır', 'alınmaz', 'sakın', 'riskli', 'tehlikeli' gibi kelimeleri KULLANMA.
  - Fiyat rakamı verme, sadece 'Uygun/Normal/Yüksek' etiketi kullan veya null bırak.
- Dil: Türkçe.
"""

SYSTEM_PROMPT_MANUAL = """
Sen 'Oto Analiz' uygulamasında KULLANICININ KENDİ ARACI için manuel analiz yapan asistansın.
Kullanıcı bazen çok az bilgi verebilir; bu durumda bile genel, bilgilendirici bir analiz üret.

Çıktı formatın NORMAL analizle aynıdır, yani:

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
- "result" alanında aracın sahibi için 3–5 paragraflık detaylı bir rehber yaz:
  - Mevcut durum
  - Bakım öncelikleri
  - Kısa / orta vadeli masraf beklentileri
  - Uzun vadede araçla ilgili tavsiyeler
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
  "example_models": ["Örnek model 1", "Örnek model 2", "...]
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
# NORMAL / PREMIUM / MANUEL ANALİZ – tek endpoint üzerinden
# ---------------------------------------------------------
@app.post("/analyze")
async def analyze(req: AnalyzeRequest) -> Dict[str, Any]:
    ctx = req.context or {}
    mode = str(ctx.get("mode", "normal")).lower()

    # Premium
    if mode == "premium":
        user_content = build_user_content(req, mode="premium")
        return call_llm_json(
            model_name=OPENAI_MODEL_PREMIUM,
            system_prompt=SYSTEM_PROMPT_PREMIUM,
            user_content=user_content,
            mode="premium",
            req=req,
        )

    # Manuel / kendi araç analizi
    if mode == "manual":
        user_content = build_user_content(req, mode="manual")
        return call_llm_json(
            model_name=OPENAI_MODEL_NORMAL,
            system_prompt=SYSTEM_PROMPT_MANUAL,
            user_content=user_content,
            mode="manual",
            req=req,
        )

    # Default: normal analiz
    user_content = build_user_content(req, mode="normal")
    return call_llm_json(
        model_name=OPENAI_MODEL_NORMAL,
        system_prompt=SYSTEM_PROMPT_NORMAL,
        user_content=user_content,
        mode="normal",
        req=req,
    )


# ---------------------------------------------------------
# Ayrı endpointler (eski client’ler için de çalışsın)
# ---------------------------------------------------------
@app.post("/premium_analyze")
async def premium_analyze(req: AnalyzeRequest) -> Dict[str, Any]:
    # Context yoksa premium olarak işaretle
    ctx = req.context or {}
    ctx["mode"] = "premium"
    req.context = ctx
    return await analyze(req)


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
        # Tamamen boş gelirse 400; Flutter tarafı buna göre kullanıcıya mesaj gösterebilir
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
