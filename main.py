import os
import base64
import json
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# .env içinden anahtarı yükle
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY bulunamadı. .env dosyasını kontrol et.")

client = OpenAI(api_key=api_key)

app = FastAPI()

# CORS – Flutter rahatça erişebilsin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # İstersen domain ile daraltırsın
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Pydantic modelleri ----------

class Profile(BaseModel):
    yearly_km: Optional[int] = None  # yıllık tahmini km
    usage: Optional[str] = None      # city / mixed / highway
    fuel_preference: Optional[str] = None  # gasoline / diesel / lpg / hybrid / electric


class Vehicle(BaseModel):
    make: Optional[str] = None       # Fiat, BMW, vs.
    model: Optional[str] = None      # Egea, 320d, vs.
    year: Optional[int] = None
    mileage_km: Optional[int] = None
    fuel: Optional[str] = None       # gasoline / diesel / lpg / hybrid / electric
    transmission: Optional[str] = None
    body_type: Optional[str] = None  # sedan / hatchback / suv ...
    price_try: Optional[float] = None  # İlanda yazan fiyat (TL)


class Context(BaseModel):
    mode: Optional[str] = None   # "normal" veya "premium"
    locale: Optional[str] = None # tr-TR vs.


class AnalyzeRequest(BaseModel):
    profile: Optional[Profile] = None
    vehicle: Optional[Vehicle] = None
    ad_description: Optional[str] = None
    screenshot_base64: Optional[str] = None
    context: Optional[Context] = None


# ---------- Segment / bakım maliyet profilleri ----------

BRAND_COST_CATEGORY: Dict[str, str] = {
    # Ucuz / düşük maliyetli (cheap)
    "fiat": "cheap",
    "renault": "cheap",
    "dacia": "cheap",
    "peugeot": "cheap",
    "citroen": "cheap",

    # Orta seviye (mid)
    "hyundai": "mid",
    "kia": "mid",
    "toyota": "mid",
    "honda": "mid",
    "opel": "mid",
    "ford": "mid",

    # Orta-yüksek (mid_high)
    "volkswagen": "mid_high",
    "skoda": "mid_high",
    "seat": "mid_high",

    # Premium / pahalı
    "bmw": "premium",
    "mercedes": "premium",
    "mercedes-benz": "premium",
    "audi": "premium",
    "volvo": "premium",
}

COST_PROFILES: Dict[str, Dict[str, Any]] = {
    "cheap": {
        "label": "Daha uygun bakım maliyeti",
        "yearly_maintenance_tl": (6000, 9000),
        "big_service_every_3y_tl": 20000,
        "tyre_set_tl": (9000, 12000),
        "kasko_rate_percent": (1.8, 3.2),
        "parts_availability_score": 5,
        "parts_price_score": 1,
        "market_speed_score": 5,
    },
    "mid": {
        "label": "Orta seviye bakım maliyeti",
        "yearly_maintenance_tl": (8000, 12000),
        "big_service_every_3y_tl": 25000,
        "tyre_set_tl": (12000, 18000),
        "kasko_rate_percent": (2.5, 4.0),
        "parts_availability_score": 4,
        "parts_price_score": 2,
        "market_speed_score": 4,
    },
    "mid_high": {
        "label": "Orta-yüksek bakım maliyeti",
        "yearly_maintenance_tl": (10000, 18000),
        "big_service_every_3y_tl": 30000,
        "tyre_set_tl": (14000, 22000),
        "kasko_rate_percent": (3.0, 5.0),
        "parts_availability_score": 3,
        "parts_price_score": 3,
        "market_speed_score": 3,
    },
    "premium": {
        "label": "Yüksek bakım maliyeti",
        "yearly_maintenance_tl": (15000, 30000),
        "big_service_every_3y_tl": 45000,
        "tyre_set_tl": (18000, 28000),
        "kasko_rate_percent": (4.0, 7.0),
        "parts_availability_score": 3,
        "parts_price_score": 5,
        "market_speed_score": 3,
    },
    "unknown": {
        "label": "Belirsiz (marka/segment net değil)",
        "yearly_maintenance_tl": (10000, 14000),
        "big_service_every_3y_tl": 24000,
        "tyre_set_tl": (12000, 18000),
        "kasko_rate_percent": (2.5, 4.5),
        "parts_availability_score": 3,
        "parts_price_score": 3,
        "market_speed_score": 3,
    },
}


def detect_cost_category(vehicle: Optional[Vehicle]) -> str:
    if not vehicle or not vehicle.make:
        return "unknown"

    make = vehicle.make.lower().strip()
    first_word = make.split()[0]  # "Fiat Egea" yazarsa ilkini al
    return BRAND_COST_CATEGORY.get(first_word, "unknown")


def build_cost_profile_text(vehicle: Optional[Vehicle]) -> str:
    category = detect_cost_category(vehicle)
    profile = COST_PROFILES.get(category, COST_PROFILES["unknown"])

    yearly_min, yearly_max = profile["yearly_maintenance_tl"]
    tyre_min, tyre_max = profile["tyre_set_tl"]
    kasko_min, kasko_max = profile["kasko_rate_percent"]
    big_service = profile["big_service_every_3y_tl"]

    lines = [
        f"Bu araç, bakım ve yedek parça maliyeti açısından '{profile['label']}' kategorisinde değerlendirilebilir.",
        f"- Yıllık rutin bakım: yaklaşık {yearly_min:,} – {yearly_max:,} TL",
        f"- 3 yılda bir gelebilecek büyük bakım paketi (triger, yürüyen vb.): yaklaşık {big_service:,} TL",
        f"- 4 adet lastik değişimi: yaklaşık {tyre_min:,} – {tyre_max:,} TL",
        f"- Tahmini kasko oranı: araç bedelinin yaklaşık %{kasko_min:.1f} – %{kasko_max:.1f} aralığında.",
        f"- Parça bulunabilirlik skoru (1–5): {profile['parts_availability_score']} (5 = çok kolay bulunur).",
        f"- Parça fiyat seviyesi (1–5): {profile['parts_price_score']} (5 = en pahalı).",
        f"- İkinci el piyasası/piyasa hızı (1–5): {profile['market_speed_score']} (5 = çok hızlı satılır).",
        "",
        "Bu tutarlar 2024–2025 dönemi Türkiye koşulları için yaklaşık değerlerdir; kesin fiyat değildir "
        "ama segmentin maliyet seviyesini anlamak için güçlü bir referans sağlar.",
    ]

    return "\n".join(lines)


# ---------- Yakıt fiyatları ve tüketim profilleri ----------

FUEL_PRICES_TL_PER_LITER: Dict[str, float] = {
    "gasoline": 52.0,
    "diesel": 53.0,
    "lpg": 29.0,
    "hybrid": 52.0,   # referans olarak benzin kullanıyoruz
    "electric": 0.0,  # lt mantığı yok
}

FUEL_CONSUMPTION_L_PER_100KM: Dict[str, Dict[str, tuple]] = {
    "cheap": {
        "city": (7.0, 8.5),
        "mixed": (5.5, 6.5),
        "highway": (4.5, 5.5),
    },
    "mid": {
        "city": (8.0, 9.0),
        "mixed": (6.5, 7.5),
        "highway": (5.0, 6.0),
    },
    "mid_high": {
        "city": (9.0, 11.0),
        "mixed": (7.5, 9.0),
        "highway": (6.0, 7.5),
    },
    "premium": {
        "city": (8.0, 10.0),
        "mixed": (6.5, 8.0),
        "highway": (5.5, 7.0),
    },
}


def get_fuel_price_tl_per_liter(fuel: Optional[str]) -> Optional[float]:
    if not fuel:
        return None
    key = fuel.lower().strip()
    return FUEL_PRICES_TL_PER_LITER.get(key)


def estimate_yearly_fuel_cost_tl(
    yearly_km: Optional[int],
    usage: Optional[str],
    fuel: Optional[str],
    cost_category: Optional[str],
) -> Optional[Dict[str, Any]]:
    if not yearly_km or yearly_km <= 0:
        return None
    if not fuel:
        return None

    fuel_price = get_fuel_price_tl_per_liter(fuel)
    if fuel_price is None or fuel_price <= 0:
        return None

    category = cost_category or "mid"
    cat_profile = FUEL_CONSUMPTION_L_PER_100KM.get(category)
    if not cat_profile:
        cat_profile = FUEL_CONSUMPTION_L_PER_100KM["mid"]

    usage_key = (usage or "mixed").lower()
    if usage_key not in cat_profile:
        usage_key = "mixed"

    cons_min, cons_max = cat_profile[usage_key]
    avg_consumption = (cons_min + cons_max) / 2.0  # lt/100km

    fuel_lower = fuel.lower()
    if fuel_lower == "hybrid":
        avg_consumption *= 0.7  # hibritte %30 daha düşük varsayımı
    if fuel_lower == "lpg":
        avg_consumption *= 1.2  # LPG'de litre bazlı tüketim artışı

    liters_per_year = yearly_km * (avg_consumption / 100.0)
    yearly_cost = liters_per_year * fuel_price
    monthly_cost = yearly_cost / 12.0

    return {
        "avg_l_per_100km": round(avg_consumption, 1),
        "liters_per_year": int(liters_per_year),
        "yearly_cost_tl": int(yearly_cost),
        "monthly_cost_tl": int(monthly_cost),
    }


def build_fuel_cost_text(req: AnalyzeRequest) -> str:
    profile = req.profile
    vehicle = req.vehicle

    if not profile or not vehicle:
        return (
            "Kullanıcının yıllık km bilgisi veya araç yakıt tipi eksik olduğu için yakıt maliyeti hesabı "
            "ancak genel seviyede yapılabilir. Yine de segmentine ve Türkiye'deki yüksek yakıt fiyatlarına göre "
            "yakıt ekonomisini yorumla."
        )

    yearly_km = profile.yearly_km
    usage = profile.usage
    fuel = vehicle.fuel

    cost_category = detect_cost_category(vehicle)
    fuel_info = estimate_yearly_fuel_cost_tl(
        yearly_km=yearly_km,
        usage=usage,
        fuel=fuel,
        cost_category=cost_category,
    )

    if not fuel_info:
        return (
            "Yakıt tüketimi ve fiyatı için net değerler hesaplanamadı. "
            "Yine de segmentine göre yakıt tüketimi ve Türkiye'deki yakıt fiyatlarını dikkate alarak "
            "genel olarak ekonomik olup olmadığını değerlendir."
        )

    usage_label_map = {
        "city": "şehir içi ağırlıklı",
        "highway": "uzun yol ağırlıklı",
        "mixed": "karışık kullanım",
    }
    usage_label = usage_label_map.get((usage or "mixed").lower(), "karışık kullanım")

    fuel_price = get_fuel_price_tl_per_liter(fuel) or 0.0

    return (
        "Yakıt maliyeti tahmini (Türkiye 2025 sonu fiyat seviyelerine göre, yaklaşık):\n"
        f"- Kullanım profili: {usage_label}\n"
        f"- Yakıt tipi: {fuel}\n"
        f"- Yaklaşık litre fiyatı: {fuel_price:.1f} TL\n"
        f"- Ortalama tüketim: {fuel_info['avg_l_per_100km']} lt/100 km\n"
        f"- Yıllık km: {yearly_km} km\n"
        f"- Yıllık tahmini yakıt tüketimi: {fuel_info['liters_per_year']} litre\n"
        f"- Yıllık tahmini yakıt maliyeti: yaklaşık {fuel_info['yearly_cost_tl']:,} TL\n"
        f"- Aylık tahmini yakıt maliyeti: yaklaşık {fuel_info['monthly_cost_tl']:,} TL\n\n"
        "Bu rakamlar kesin değildir, yaklaşık değerlerdir; analiz yaparken bütçe uyumu, bölgesel yakıt fiyatı farkları "
        "ve gerçek kullanım tarzını da mutlaka göz önünde bulundur."
    )


# ---------- Model bazlı kronik sorunlar / dikkat noktaları ----------

MODEL_ISSUES: Dict[str, Dict[str, Any]] = {
    "fiat_egea": {
        "title": "Fiat Egea (tüm nesiller)",
        "segment_category": "cheap",
        "summary": (
            "Türkiye'nin en çok satan modellerinden. Uygun fiyat, yaygın servis ve bol yedek parça sayesinde "
            "özellikle filo ve bütçe odaklı kullanıcılar için mantıklı bir tercih."
        ),
        "known_issues": [
            "Bazı kullanıcılarda direksiyon kutusu / milinden gelen ses ve boşluk şikâyetleri görülüyor; "
            "yüksek kilometrede revizyon gerekebiliyor.",
            "1.6 Multijet dizellerde yüksek km'de enjektör, turbo ve EGR kaynaklı masraf ihtimali artabiliyor.",
            "1.4 benzin + LPG kombinasyonunda yağ eksiltme ve supap aşınması şikâyetleri olabiliyor; LPG montaj kalitesi kritik.",
        ],
        "buy_tips": [
            "Filo çıkması araçlarda km düşürme ve yoğun şehir içi kullanım nedeniyle yıpranma riskine dikkat et.",
            "Direksiyon tam sağ/sol yapılıp boşluk ve ses kontrolü yapılmalı; yürüyen aksam lifte kaldırılıp incelenmeli.",
            "LPG'li araçta montaj faturası, marka-model ve regüler ayar kayıtları mutlaka görülmeli.",
        ],
    },
    "renault_clio": {
        "title": "Renault Clio (B segment, özellikle 4–5)",
        "segment_category": "cheap",
        "summary": (
            "Türkiye'de en çok satan B segment modellerden. Parça ve servis ağı yaygın, işletme maliyeti düşük. "
            "Şehir içi ve ilk araç için mantıklı."
        ),
        "known_issues": [
            "TCe turbo benzinli motorlarda uzun vadede turbo ve emme hattında kurum birikimi kaynaklı performans düşüşü "
            "ve ufak arızalar görülebiliyor.",
            "İç trim kalitesi nedeniyle kapı içi ve konsol bölgesinde trim gıcırtıları sık rapor ediliyor.",
        ],
        "buy_tips": [
            "Turbo benzinli motorlarda yağ değişim aralığının kısaltılmış olması (10 bin km civarı) olumlu.",
            "Test sürüşünde süspansiyon ve trim seslerine özellikle bozuk yolda dikkat et.",
            "Düzenli bakım geçmişi olan, kazası-boyası net araçları tercih etmek önemli.",
        ],
    },
    "renault_megane4_sedan": {
        "title": "Renault Megane 4 Sedan",
        "segment_category": "mid",
        "summary": (
            "Türkiye'de çok satan C sınıfı sedan. Özellikle 1.5 dCi + EDC kombinasyonu düşük yakıt tüketimiyle biliniyor; "
            "donanım/fiyat dengesi güçlü."
        ),
        "known_issues": [
            "1.5 dCi motor genel olarak sağlam; ancak yüksek km'de turbo, enjektör, DPF tıkanması ve debriyaj seti masrafları "
            "görülebiliyor.",
            "Kullanıcı şikâyetlerinde arka koltuk ve bagaj çevresine su alma problemleri ve trim sesleri dikkat çekiyor.",
            "Elektronik donanım (multimedya, sensörler, dijital gösterge) tarafında yazılımsal ufak arızalar rapor edilebiliyor.",
        ],
        "buy_tips": [
            "EDC şanzıman sıcak-soğuk test edilmeli; kalkışta titreme, vites geçişinde vuruntu olmamalı.",
            "Bagaj havuzu ve arka koltuk altı nem/ıslaklık açısından kontrol edilmeli.",
            "Dizel araçta DPF durumu, egzoz dumanı ve turbo sesi mutlaka uzman bir ekspertizle incelenmeli.",
        ],
    },
    "toyota_corolla_1_6": {
        "title": "Toyota Corolla 1.6 (2010–2023 civarı)",
        "segment_category": "mid",
        "summary": (
            "Dayanıklılığı ve ikinci elde değerini iyi korumasıyla bilinen C sedan. Doğru bakım ile uzun ömürlü ve "
            "sorunsuz kullanım sunma potansiyeli yüksek."
        ),
        "known_issues": [
            "Bazı eski nesil benzinli motorlarda yüksek kilometrede yağ tüketimi artışı rapor edilmiş; piston segman ve "
            "valf keçeleri ile ilişkilendiriliyor.",
            "Nadir de olsa valf zamanlama / değişken supap kontrol modülü arızaları görülebiliyor.",
            "Klima yönlendirme klapeleri ve iç plastik trimlerde ufak sorunlar rapor edilebiliyor.",
        ],
        "buy_tips": [
            "Yağ eksiltme var mı; karter altı, egzoz dumanı ve bujiler kontrol ettirilmeli.",
            "Aracın periyodik bakımlarının zamanında ve kaliteli yağ ile yapılmış olması önemli.",
            "LPG'li araçlarda montaj kalitesi, regüler bakım ve supap ayar geçmişi özellikle sorulmalı.",
        ],
    },
    "honda_civic_fc5": {
        "title": "Honda Civic FC5 Sedan (2016–2021)",
        "segment_category": "mid_high",
        "summary": (
            "Türkiye'de özellikle LPG'li sedan olarak çok popüler. Sürüş hissi ve tasarımı beğeniliyor; ancak boya ve "
            "CVT şanzımanla ilgili kronik şikâyetler var."
        ),
        "known_issues": [
            "Tavan ve C sütunu bölgelerinde vernik/boya kalkması, pek çok kullanıcıda kronik problem olarak rapor ediliyor.",
            "Bazı araçlarda CVT şanzıman arızaları ve yüksek onarım maliyetleri söz konusu; yağ değişimi ihmal edilmemeli.",
            "Fabrikasyon LPG'li Eco versiyonlarında LPG seviye sensörü ve kalibrasyon problemleri yaşanabiliyor.",
            "Kapı kilit mekanizması, su eksiltme ve çeşitli trim sesleriyle ilgili şikâyetler mevcut.",
        ],
        "buy_tips": [
            "Tavan, kapı sütunları ve bagaj çevresi boya kalınlığı ve vernik atması açısından detaylı kontrol edilmeli.",
            "Test sürüşünde CVT'nin yokuş performansına, kaydırma yapıp yapmadığına, titreme ve uğultu sesine dikkat edilmeli.",
            "LPG sisteminde arıza kaydı, sensör değişimi ve kalibrasyon geçmişi sorulmalı.",
        ],
    },
    "dacia_duster": {
        "title": "Dacia Duster (2010–2024)",
        "segment_category": "mid",
        "summary": (
            "Uygun fiyatlı, basit yapılı ve arazi kabiliyeti olan kompakt SUV. Mekanik yapı genel olarak sağlam, ancak "
            "yürüyen ve egzoz tarafında bazı kronik noktalar rapor edilmiş."
        ),
        "known_issues": [
            "Kullanıcı verilerinde şanzıman / vites kutusu sorunları, DPF tıkanması ve arka dingil sesleri öne çıkıyor.",
            "İlk nesillerde ön takımda iz ayarı / sola çekme problemleri ve bazı paslanma şikâyetleri görülmüş.",
            "Küçük turbo benzin motorlarında soğutma ve contalarla ilgili ufak kaçak şikâyetleri olabiliyor.",
        ],
        "buy_tips": [
            "Test sürüşünde direksiyonu serbest bırakıp aracın sağ/sol çekip çekmediğine bak; rot-balans mutlaka kontrol edilmeli.",
            "Dizel versiyonlarda DPF rejenerasyon geçmişi, egzoz dumanı ve turbo sesi incelenmeli.",
            "Arazi kullanımı görmüş araçlarda alt karter, şasi ve diferansiyel darbeye karşı detaylı ekspertiz şart.",
        ],
    },
    "vag_c_segment_tsi_dsg": {
        "title": "VW / Skoda / Seat C-Segment TSI + 7 ileri DSG",
        "segment_category": "mid_high",
        "summary": (
            "Golf, Octavia, Leon vb. C segment VAG araçlarda kullanılan 1.2–1.5 TSI motor + 7 ileri kuru kavrama DSG "
            "kombinasyonu; sürüş keyfi yüksek ancak DSG ve bazı eski TSI serilerinde kronik sorun geçmişi var."
        ),
        "known_issues": [
            "7 ileri kuru kavrama DQ200 DSG'de mekatronik arızası, kavrama aşınması, düşük hızda titreme ve vuruntu şikâyetleri sık görülüyor.",
            "Bazı eski TSI motorlarda zamanlama zinciri ve yağ tüketimi sorunları geçmişte sıkça rapor edilmiş; düzgün bakımlı araçlarda risk azalıyor.",
        ],
        "buy_tips": [
            "DSG'li araçlarda kalkış ve düşük hız test sürüşü çok önemli; titreme, vuruntu, gecikmeli vites geçişi varsa uzak dur.",
            "Şanzıman yağ/bakım kayıtları ve varsa mekatronik/kavrama değişim faturaları mutlaka istenmeli.",
            "Garantisi devam eden veya yakın zamanda şanzıman revizyonu yapılmış araçlar daha güvenli tercihtir.",
        ],
    },
    "bmw_3series_n47": {
        "title": "BMW 3 Serisi N47 Dizel (2007–2014 civarı)",
        "segment_category": "premium",
        "summary": (
            "N47 dizel motor, performansı ve verimliliği ile seviliyor; ancak zamanlama zinciri sorunları nedeniyle "
            "dikkat edilmesi gereken bir motor ailesi olarak biliniyor."
        ),
        "known_issues": [
            "Zamanlama zinciri motorun arka tarafında; arıza durumunda motorun indirilmesi gerekebiliyor ve bu yüksek işçilik masrafı demek.",
            "Zincir gergisi ve plastik kızakların yıpranması sonucu rölantide ve ilk çalıştırmada karakteristik 'zincir sesi' ortaya çıkabiliyor; "
            "ihmal edilirse ağır motor hasarı riski var.",
        ],
        "buy_tips": [
            "İlk çalıştırmada kaputu açıp zincir sesi için dikkatlice dinle; soğukken gelen metalik tıkırtılar ciddi uyarı işareti.",
            "Zincir değişimi yapılmışsa kullanılan parçalar (orijinal/OEM) ve işçilik faturası mutlaka görülmeli.",
            "Yüksek km'li N47 alırken zincir masrafını bütçeye ekstra risk kalemi olarak eklemek mantıklı.",
        ],
    },
}


def get_model_issue_keys(vehicle: Optional[Vehicle]) -> List[str]:
    if not vehicle or not vehicle.make or not vehicle.model:
        return []

    make = vehicle.make.lower()
    model = vehicle.model.lower()
    keys: List[str] = []

    if "fiat" in make and "egea" in model:
        keys.append("fiat_egea")

    if "renault" in make and "clio" in model:
        keys.append("renault_clio")

    if "renault" in make and "megane" in model:
        keys.append("renault_megane4_sedan")

    if "toyota" in make and "corolla" in model:
        keys.append("toyota_corolla_1_6")

    if "honda" in make and "civic" in model:
        keys.append("honda_civic_fc5")

    if "dacia" in make and "duster" in model:
        keys.append("dacia_duster")

    if make in ("volkswagen", "skoda", "seat") and any(
        x in model for x in ["golf", "octavia", "leon", "jetta"]
    ):
        keys.append("vag_c_segment_tsi_dsg")

    if "bmw" in make and any(x in model for x in ["320", "318", "520", "x1", "x3"]):
        keys.append("bmw_3series_n47")

    return keys


def build_model_insights_text(vehicle: Optional[Vehicle]) -> str:
    keys = get_model_issue_keys(vehicle)
    if not keys:
        return (
            "Bu araç için elimizde özel model/şanzıman kronik sorun verisi bulunmuyor. "
            "Genel segment maliyeti, bakım profili ve ekspertiz raporuna göre yorum yap."
        )

    parts: List[str] = []
    for key in keys:
        info = MODEL_ISSUES.get(key)
        if not info:
            continue

        parts.append(f"Model özel notlar – {info['title']}:")
        parts.append(f"- Genel özet: {info['summary']}")

        if info.get("known_issues"):
            parts.append("- Bilinen kronik / sık görülen sorunlar:")
            for issue in info["known_issues"]:
                parts.append(f"  • {issue}")

        if info.get("buy_tips"):
            parts.append("- Bu aracı alırken dikkat edilmesi önerilen noktalar:")
            for tip in info["buy_tips"]:
                parts.append(f"  • {tip}")

        parts.append("")  # boş satır

    return "\n".join(parts)


# ---------- Prompt helper'ları ----------

def build_system_prompt(mode: str) -> str:
    base = (
        "Sen Türkiye ikinci el binek araç piyasasını iyi bilen, dürüst, net konuşan bir oto danışmansın. "
        "Sana kullanıcının araç bilgileri, profil bilgileri, tahmini bakım ve yakıt maliyeti ile "
        "bazı model/şanzıman kronik sorun notları verilecek.\n\n"
        "Görevin:\n"
        "- Aracın mantıklı bir tercih olup olmadığını dürüstçe değerlendirmek,\n"
        "- Kullanıcının kullanım profiline ve bütçesine göre artı/eksi yönleri anlatmak,\n"
        "- Bakım maliyeti, yakıt masrafı, parça bulunabilirlik ve ikinci el piyasasını dengeleyerek yorum yapmak,\n"
        "- Gereksiz teknik terimlere boğmadan, ama boş da konuşmadan net ve anlaşılır olmak.\n\n"
        "Cevabın mutlaka geçerli JSON formatında olmalı. Aşağıdaki şemaya uymalısın:\n"
        "{\n"
        '  "scores": {\n'
        '    "overall_100": number,  // 0–100, ne kadar mantıklı bir tercih\n'
        '    "price_score_100": number,  // fiyat uygunluğu\n'
        '    "maintenance_cost_score_100": number,  // bakım/mekanik maliyet\n'
        '    "fuel_cost_score_100": number,  // yakıt ekonomisi\n'
        '    "risk_score_100": number  // beklenmedik masraf ve kronik sorun riski; yüksek = riskli\n'
        "  },\n"
        '  "summary": {\n'
        '    "short_text": string,  // 2-3 cümlede genel sonuç\n'
        '    "long_text": string,   // daha detaylı açıklama\n'
        '    "pros": [string],      // artılar listesi\n'
        '    "cons": [string],      // eksiler listesi\n'
        '    "warnings": [string]   // özellikle dikkat edilmesi gereken uyarılar\n'
        "  }\n"
        "}\n\n"
        "Sadece JSON üret; açıklama metni JSON'un içindeki alanlarda olsun, JSON dışında ek cümle yazma."
    )

    if (mode or "").lower() == "premium":
        extra = (
            "\n\nBu analiz PREMIUM modda. Daha detaylı ol:\n"
            "- 'long_text' içinde en az 8–12 cümlelik kapsamlı bir değerlendirme yap,\n"
            "- Bakım ve yakıt maliyeti rakamlarını yorumlayarak fiyat/performans analizi yap,\n"
            "- Elindeki model özel kronik sorunları mutlaka değerlendirmeye kat,\n"
            "- Kullanıcının tipik profiline göre (şehir içi/uzun yol/yıllık km) gerçekçi senaryolar ver."
        )
    else:
        extra = (
            "\n\nBu analiz NORMAL modda. Daha kısa, sade ama yine de dolu ol:\n"
            "- 'long_text' 5–8 cümle civarı olsun,\n"
            "- Kullanıcıyı korkutmadan, ama gerçek riskleri de saklamadan anlat,\n"
            "- Rakamları yuvarlayarak ve 'yaklaşık' diyerek kullan."
        )

    return base + extra


def build_user_text(req: AnalyzeRequest) -> str:
    v = req.vehicle
    p = req.profile

    lines: List[str] = []
    lines.append("Kullanıcının verdiği temel bilgiler aşağıdadır.")
    lines.append("")
    lines.append("ARAÇ BİLGİLERİ:")

    if v:
        lines.append(f"- Marka: {v.make or 'bilinmiyor'}")
        lines.append(f"- Model: {v.model or 'bilinmiyor'}")
        lines.append(f"- Yıl: {v.year or 'bilinmiyor'}")
        lines.append(f"- Kilometre: {v.mileage_km or 'bilinmiyor'}")
        lines.append(f"- Yakıt tipi: {v.fuel or 'bilinmiyor'}")
        lines.append(f"- Vites: {v.transmission or 'bilinmiyor'}")
        lines.append(f"- Kasa tipi: {v.body_type or 'bilinmiyor'}")
        lines.append(f"- İlanda yazan fiyat (TL): {v.price_try or 'bilinmiyor'}")
    else:
        lines.append("- Araç bilgisi gönderilmemiş.")

    lines.append("")
    lines.append("KULLANICI PROFİLİ:")

    if p:
        lines.append(f"- Yıllık tahmini km: {p.yearly_km or 'bilinmiyor'}")
        lines.append(f"- Kullanım tipi: {p.usage or 'bilinmiyor'} (city/mixed/highway)")
        lines.append(f"- Yakıt tercihi: {p.fuel_preference or 'belirtmemiş'}")
    else:
        lines.append("- Kullanıcı profil bilgisi gönderilmemiş.")

    if req.ad_description:
        lines.append("")
        lines.append("İLAN AÇIKLAMASI (satıcının metni):")
        lines.append(req.ad_description)

    if req.screenshot_base64:
        lines.append("")
        lines.append(
            "Ek olarak, araç ilanının ekran görüntüsü de veriliyor (fiyat, donanım, görsel durum vb. görselden okunabilir). "
            "Görseldeki yazıları ve genel durumu da analizine kat."
        )

    lines.append("")
    lines.append(
        "Bu bilgiler ışığında; aracın alınabilirliği, fiyatı, beklenen bakım/yakıt maliyetleri ve olası riskler hakkında "
        "gerçekçi, mantıklı ve Türkiye piyasasına uygun bir değerlendirme yap."
    )

    return "\n".join(lines)


# ---------- /analyze endpoint'i ----------

@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    try:
        mode = (req.context.mode if req.context and req.context.mode else "normal") if req.context else "normal"
    except Exception:
        mode = "normal"

    system_prompt = build_system_prompt(mode)

    core_text = build_user_text(req)
    cost_text = build_cost_profile_text(req.vehicle)
    fuel_text = build_fuel_cost_text(req)
    model_insights_text = build_model_insights_text(req.vehicle)

    combined_user_text = (
        core_text
        + "\n\n---\n\n"
        + "BAKIM / PARÇA / PİYASA PROFİLİ:\n"
        + cost_text
        + "\n\n---\n\n"
        + "YAKIT TÜKETİMİ VE YAKIT MALİYETİ:\n"
        + fuel_text
        + "\n\n---\n\n"
        + "MODEL / ŞANZIMAN ÖZEL KRONİK SORUNLAR ve DİKKAT NOKTALARI:\n"
        + model_insights_text
    )

    # LLM'e gidecek user content (metin + varsa görsel)
    user_content: List[Dict[str, Any]] = []

    # Screenshot geldiyse input_image olarak ekle
    if req.screenshot_base64:
        try:
            base64.b64decode(req.screenshot_base64)
            user_content.append({
                "type": "input_image",
                "image_base64": req.screenshot_base64,
            })
        except Exception:
            # Görsel bozuksa yok sayıp sadece metinle devam edelim
            pass

    user_content.append({
        "type": "input_text",
        "text": combined_user_text,
    })

    # JSON şemamız
    analysis_schema = {
        "name": "VehicleAnalysis",
        "schema": {
            "type": "object",
            "properties": {
                "scores": {
                    "type": "object",
                    "properties": {
                        "overall_100": {"type": "number"},
                        "price_score_100": {"type": "number"},
                        "maintenance_cost_score_100": {"type": "number"},
                        "fuel_cost_score_100": {"type": "number"},
                        "risk_score_100": {"type": "number"},
                    },
                    "required": ["overall_100"],
                    "additionalProperties": True,
                },
                "summary": {
                    "type": "object",
                    "properties": {
                        "short_text": {"type": "string"},
                        "long_text": {"type": "string"},
                        "pros": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "cons": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "warnings": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["short_text"],
                    "additionalProperties": True,
                },
            },
            "required": ["scores", "summary"],
            "additionalProperties": True,
        },
        "strict": False,
    }

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {
                    "role": "system",
                    "content": [
                        {"type": "input_text", "text": system_prompt},
                    ],
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": analysis_schema,
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM isteği başarısız: {e}")

    try:
        # Responses API'de ilk output'un ilk content'inden text'i alıyoruz
        output = response.output[0].content[0].text  # type: ignore[attr-defined]
        data = json.loads(output)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Modelden beklenen JSON formatı alınamadı: {e}",
        )

    return data


@app.get("/")
async def root():
    return {"status": "ok", "message": "Oto Analiz backend çalışıyor"}
