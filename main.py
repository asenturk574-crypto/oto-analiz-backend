import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

# OpenAI client (key ortam değişkeninden geliyor)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(
    title="Oto Analiz Backend",
    description="2. el araç ilanı için normal ve premium analiz servisi",
    version="1.0.0",
)

# Flutter'dan rahat erişim için CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # istersen ileride kendi domainine sınırlarız
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    url: str
    budget: str


def build_prompt(url: str, budget: str, premium: bool) -> str:
    """
    Normal ve premium için farklı detay seviyesinde prompt oluşturur.
    """
    base = f"""
Sen Türkiye'de ekspertiz raporu hazırlayan çok deneyimli bir oto eksperisin.
Kullanıcı sana bir 2. el araç ilan linki ve bütçesini gönderiyor.

İlan URL: {url}
Kullanıcının bütçesi: {budget} TL

Bu ilanın içeriğini göremiyor olsan bile, Türkiye'deki ikinci el piyasası, 
tipik kronik problemler, kilometre riskleri ve pazarlık payı gibi konularda 
genel fakat MANTIKLI bir analiz yapacaksın.
"""

    if not premium:
        # NORMAL (ÜCRETSİZ) ANALİZ
        return base + """
KISA VE ÖZ bir analiz yap. Çıktıyı Türkçe ver ve şu formatta yaz:

- Tahmini fiyat uygunluğu: (kısaca yorumla, örneğin "Piyasanın biraz üstünde", "Fena değil", "Gayet uygun")
- Temel risk yorumu: (örneğin "Kilometreye dikkat edilmeli", "Model kronik sorunları açısından kontrol şart")
- Basit avantajlar (en fazla 3 madde)
- Basit dezavantajlar (en fazla 3 madde)
- Sonuç: (Alınır / Sınırda / Alınmaz) şeklinde tek cümlelik karar ver.

Kendinden uydurma net rakamlar yazma, ama "genel piyasa" üzerinden mantıklı yorum yap.
KISA TUT, maksimum 10-15 cümle olsun.
"""
    else:
        # PREMIUM ANALİZ
        return base + """
DETAYLI bir "oto ekspertiz raporu" hazırla. Çıktıyı Türkçe ver.
Aşağıdaki başlıkları mutlaka sırayla kullan ve BÜTÜN METNİ TEK BİR METİN olarak ver:

1) Araç Hakkında Genel İzlenim:
- Olası segmenti, kullanım amacı (şehir içi, ticari, aile aracı vb.)
- Bütçeye göre genel ilk izlenim

2) Fiyat / Performans Analizi:
- Piyasa ortalamasına göre fiyat tahmini (ucuz/normal/pahalı şeklinde)
- Pazarlık payı yorumun
- Bütçe ile ne kadar uyumlu (yüzde tahmini verebilirsin, örn: "%80 uygun")

3) Olası Kronik Sorunlar ve Risk Noktaları:
- Bu tip araçlarda sık görülen kronik sorunlar (varsa)
- Yaş/kilometre arttıkça çıkabilecek tipik problemler
- Mutlaka kontrol edilmesi gereken noktalar (en az 4-5 madde)

4) Motor ve Mekanik Durum Tahmini:
- Bakım kayıtları, yağ değişimi, turbo, enjektör, şanzıman vb. hakkında genel değerlendirme (tahmini)
- Şehir içi/uzun yol kullanımına etkisi

5) Gövde, Kaza ve Boya İhtimali:
- Muhtemel kaza geçmişi riskleri
- Değişen/boyalı parça ihtimali hakkında genel yorum
- Airbag, şasi, direk kontrolünün önemi

6) KM ve Kullanım Şekli Yorumu:
- Kilometre ile oynama ihtimali (genel piyasaya göre yorum yap)
- Kullanım şekline göre yıpranma tahmini

7) Avantajlar:
- En az 4 net madde yaz (örneğin: "Parça bulunabilirliği iyi", "Yakıt tüketimi makul", "İkinci eli hızlı satılır" vb.)

8) Dezavantajlar:
- En az 4 net madde yaz (örneğin: "Modelde kronik şanzıman sorunları görülebiliyor" vb.)

9) Risk Skoru (%):
- 0 ile 100 arasında bir RİSK skoru ver (0 = çok güvenli, 100 = çok riskli)
- Kısa bir cümleyle bu skoru açıkla.

10) Son Karar (Alınır / Sınırda / Alınmaz):
- Tek cümlede net konuş: "Bu araç genel olarak ALINIR." gibi
- Yanına kısa bir gerekçe ekle.

Notlar:
- Tamamen uydurma detaylı hikâyeler yazma, ama Türkiye'deki ikinci el araç piyasası mantığına uygun, 
profesyonel bir yorum yap.
- Kullanıcıya dürüst ve net ol.
"""

def call_openai(prompt: str) -> str:
    """
    OpenAI ile sohbet tamamlaması alır ve metni döndürür.
    """
    chat_completion = client.chat.completions.create(
        model="gpt-4o-mini",  # istersen "gpt-4.1-mini" vb. kullanabilirsin
        messages=[
            {
                "role": "system",
                "content": "Sen Türkiye'de çalışan, dürüst ve detaycı bir oto ekspertiz uzmansın."
            },
            {
                "role": "user",
                "content": prompt
            },
        ],
        temperature=0.6,
    )

    return chat_completion.choices[0].message.content


@app.post("/analyze")
async def analyze_basic(body: AnalyzeRequest):
    """
    ÜCRETSİZ / NORMAL analiz:
    Kısa ve özet bir değerlendirme döner.
    """
    prompt = build_prompt(body.url, body.budget, premium=False)
    try:
        analysis = call_openai(prompt)
        return {"analysis": analysis, "mode": "basic"}
    except Exception as e:
        # Hata durumunda kullanıcıya anlamlı bir mesaj dön
        return {
            "analysis": f"Analiz sırasında bir hata oluştu: {e}",
            "mode": "basic",
            "error": True,
        }


@app.post("/analyze_premium")
async def analyze_premium(body: AnalyzeRequest):
    """
    PREMIUM analiz:
    Detaylı ekspertiz raporu formatında çıktı döner.
    """
    prompt = build_prompt(body.url, body.budget, premium=True)
    try:
        analysis = call_openai(prompt)
        return {"analysis": analysis, "mode": "premium"}
    except Exception as e:
        return {
            "analysis": f"Premium analiz sırasında bir hata oluştu: {e}",
            "mode": "premium",
            "error": True,
        }


@app.get("/")
async def root():
    return {"status": "ok", "message": "Oto Analiz Backend çalışıyor."}
