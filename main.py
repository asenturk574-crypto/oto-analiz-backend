from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from openai import OpenAI
import os

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class AnalyzeRequest(BaseModel):
    url: Optional[str] = ""
    usage: Optional[str] = ""
    description: Optional[str] = ""
    budget: Optional[str] = None
    city: Optional[str] = ""
    note: Optional[str] = ""
    mode: Optional[str] = "normal"   # <<< DEFAULT NORMAL

@app.post("/analyze")
async def run_analysis(request: AnalyzeRequest):

    # -------------------------------------------------
    # NORMAL TEMPLATE (Kısa – sade – kullanıcıya özel)
    # -------------------------------------------------
    normal_prompt = f"""
Kısa ve sade bir ikinci el araç analizi oluştur.

Kullanıcı Bilgileri:
- Kullanım amacı: {request.usage}
- Açıklama: {request.description}
- Bütçe: {request.budget}
- Şehir: {request.city}
- Ek notlar: {request.note}

Kurallar:
- Analiz 6–8 kısa bölümden oluşsun
- Her bölüm maximum 2 cümle
- Kullanıcıya göre yorum yap
- Premium kadar detay verme
- Maddeleri kısa tut
- Teknik derinlik yok, sade konuş

Format:
1) Araç Özeti (1–2 cümle)
2) Kullanıcı Profiline Uygunluk (1–2 cümle)
3) Motor – Yakıt Yorumu (1–2 cümle)
4) Şanzıman Yorumu (1 cümle)
5) Artılar (2 madde)
6) Eksiler (2 madde)
7) Fiyat Yorumu (1 cümle)
8) Sonuç (tek net cümle)
    """

    # -------------------------------------------------
    # PREMIUM TEMPLATE (Uzun – derin – detaylı)
    # -------------------------------------------------
    premium_prompt = f"""
Detaylı, profesyonel ve tamamen kullanıcıya özel bir premium ikinci el araç analizi oluştur.

Kullanıcı Bilgileri:
- Kullanım amacı: {request.usage}
- Açıklama: {request.description}
- Bütçe: {request.budget}
- Şehir: {request.city}
- Ek notlar: {request.note}

Kurallar:
- 12–15 başlık olsun
- Her başlık 3–5 cümle olsun
- Motor, şanzıman, kronik sorunlar, masraf, piyasa, uygunluk, alternatif araçlar vb.
- Kullanıcının kullanım senaryosuna özel yorum yap
- Premium analiz yine çok dolu ve detaylı olsun

Format tamamen premium seviyesinde olsun.
    """

    # Hangi mod?
    prompt = normal_prompt
    if request.mode == "premium":
        prompt = premium_prompt

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        return {
            "mode": request.mode,
            "analysis": response.choices[0].message.content
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analiz sırasında hata oluştu: {e}")
