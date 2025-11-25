import os
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------
#  SCRAPER – URL’den otomatik bilgi çekme
# ------------------------------

def scrape_listing(url: str):
    data = {}

    try:
        headers = {
            "User-Agent": "Mozilla/5.0"
        }
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        print("Scrape hata:", e)
        return data

    soup = BeautifulSoup(resp.text, "html.parser")

    # Fiyat
    price = soup.select_one(".classifiedInfo .price") or soup.select_one(".price")
    if price:
        raw = price.get_text(strip=True)
        digits = "".join([c for c in raw if c.isdigit()])
        if digits:
            data["price"] = float(digits)
        data["currency"] = "TRY"

    # Başlık
    title_el = soup.select_one("h1")
    if title_el:
        data["title"] = title_el.get_text(strip=True)

    # Açıklama
    desc_el = soup.select_one("#classifiedDescription") or soup.select_one(".description")
    if desc_el:
        data["description"] = desc_el.get_text(" ", strip=True)

    # KM / YIL / YAKIT / VİTES (sahibinden için tablo parse)
    table_rows = soup.select(".classifiedInfoList li")
    for row in table_rows:
        text = row.get_text(" ", strip=True).lower()
        if "km" in text:
            digits = "".join([c for c in text if c.isdigit()])
            if digits:
                data["km"] = int(digits)
        if "model" in text:
            digits = "".join([c for c in text if c.isdigit()])
            if digits:
                data["year"] = int(digits)
        if "yakıt" in text:
            if "dizel" in text:
                data["fuel"] = "Dizel"
            elif "benzin" in text:
                data["fuel"] = "Benzin"
        if "vites" in text:
            if "otomatik" in text:
                data["gear"] = "Otomatik"
            elif "manuel" in text:
                data["gear"] = "Manuel"

    return data



# ------------------------------
#   API REQUEST MODELLERİ
# ------------------------------

class AnalyzeRequest(BaseModel):
    url: Optional[str] = None
    user_budget: Optional[float] = None

    title: Optional[str] = None
    price: Optional[float] = None
    currency: Optional[str] = "TRY"
    year: Optional[int] = None
    km: Optional[int] = None
    fuel: Optional[str] = None
    gear: Optional[str] = None
    body_type: Optional[str] = None
    city: Optional[str] = None
    description: Optional[str] = None

    is_premium: bool = False


class AnalyzeResponse(BaseModel):
    analysis: str



@app.get("/")
async def root():
    return {"message": "Oto Analiz backend çalışıyor."}



# ------------------------------
# ANALİZ ENDPOINT
# ------------------------------

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_car(data: AnalyzeRequest):

    # 🔥 URL geldiyse ve diğer bilgiler boşsa otomatik scrape et
    if data.url:
        scraped = scrape_listing(data.url)
        for key, value in scraped.items():
            # Eğer kullanıcı manuel girmemişse scrape’den geleni doldur
            if getattr(data, key, None) in (None, "", 0):
                setattr(data, key, value)

    # ------------------------------
    # İLAN METNİ OLUŞTUR
    # ------------------------------
    ilan = []

    if data.title: ilan.append(f"Başlık: {data.title}")
    if data.price: ilan.append(f"Fiyat: {data.price} {data.currency}")
    if data.year: ilan.append(f"Model Yılı: {data.year}")
    if data.km: ilan.append(f"Kilometre: {data.km}")
    if data.fuel: ilan.append(f"Yakıt: {data.fuel}")
    if data.gear: ilan.append(f"Vites: {data.gear}")
    if data.body_type: ilan.append(f"Segment: {data.body_type}")
    if data.city: ilan.append(f"Şehir: {data.city}")
    if data.description: ilan.append(f"Açıklama: {data.description}")

    ilan_metni = "\n".join(ilan) if ilan else "İlan bilgisi yok."

    # Kullanıcı bütçe bilgisi
    butce = (
        f"{data.user_budget} {data.currency}"
        if data.user_budget else "Belirtilmemiş"
    )

    premium = "EVET" if data.is_premium else "HAYIR"


    # ------------------------------
    # PROMPT
    # ------------------------------

    system_prompt = """
Sen Türkiye'deki 2.el araç piyasasını çok iyi bilen kesin bir ekspertiz uzmanısın.
FİYAT UYDURMA. Sana gelen fiyatı aynen kullan.
"""

    if data.is_premium:
        user_prompt = f"""
Aşağıdaki ilanı premium detayda analiz et.

Bütçe: {butce}
Premium: {premium}

İLAN:
{ilan_metni}

KURALLAR:
- İLAN FİYATINI ASLA DEĞİŞTİRME.
- KENDİNCE YENİ FİYAT UYDURMA.
- MASRAF TAHMİNİ YAPABİLİRSİN AMA İLAN FİYATINI DEĞİŞTİRME.

FORMAT:
1) Kısa Özet
2) Olumlu Yönler
3) Riskler / Masraflar
4) Kronik Sorunlar
5) Fiyat & Piyasa Analizi
6) Pazarlık Payı Tahmini
7) Ekspertizde Baktırılacak Noktalar
8) Son Karar (AL / DÜŞÜN / UZAK DUR)
"""
    else:
        user_prompt = f"""
Aşağıdaki ilanı hızlı analiz et.

Bütçe: {butce}

İLAN:
{ilan_metni}

FİYATI DEĞİŞTİRME. SANA GELEN FİYAT: {data.price}

FORMAT:
1) Özet
2) Olumlu Yönler
3) Riskler
4) Bütçe Uygun mu?
5) Son Karar
"""


    # ------------------------------
    #   OPENAI ÇAĞRISI
    # ------------------------------

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.4
    )

    text = response.choices[0].message.content.strip()

    return AnalyzeResponse(analysis=text)
