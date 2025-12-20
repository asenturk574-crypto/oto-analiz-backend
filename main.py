import os
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

# Lokal geliştirme için .env yükle (Render'da Environment kullanılıyor)
load_dotenv()


# -------------------------------------------------
# FastAPI app
# -------------------------------------------------
app = FastAPI(title="oto-analiz-backend", version="1.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # İstersen domain bazlı kısıtlarsın
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------
# Pydantic modeller
# -------------------------------------------------
class AnalyzeRequest(BaseModel):
    # Metin alanları
    text: Optional[str] = None
    listing_title: Optional[str] = None
    description: Optional[str] = None
    extras: Optional[Dict[str, Any]] = None

    # Çoklu ekran görüntüsü (base64 listesi)
    screenshots_base64: Optional[List[str]] = None


class AnalyzeResponse(BaseModel):
    ok: bool
    result: str


# -------------------------------------------------
# Yardımcı fonksiyonlar
# -------------------------------------------------
def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Bu hata sadece backend yanlış konfigüre ise çıkar
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY bulunamadı. Render Environment bölümüne ekle.",
        )
    return OpenAI(api_key=api_key)


def build_user_text(req: AnalyzeRequest) -> str:
    """Metin alanlarını tek bir açıklama stringine çevirir."""
    parts: List[str] = []

    if req.text:
        parts.append(req.text)

    if req.listing_title:
        parts.append(f"İlan başlığı:\n{req.listing_title}")

    if req.description:
        parts.append(f"İlan açıklaması:\n{req.description}")

    if req.extras:
        parts.append(f"Ek bilgiler:\n{req.extras}")

    if not parts:
        raise HTTPException(
            status_code=400,
            detail="Boş istek. En azından 'text' veya 'listing_title/description' gönder.",
        )

    # Kullanıcıyı yönlendiren ek açıklama
    parts.append(
        "\nLütfen bu aracı Türkiye ikinci el piyasasına göre değerlendir: "
        "kronik sorun riskleri, olası masraflar, artı/eksi yönler ve pazarlık tavsiyesini "
        "detaylı ama anlaşılır bir Türkçe ile anlat."
    )

    return "\n\n".join(parts)


def build_user_content(req: AnalyzeRequest) -> List[Dict[str, Any]]:
    """
    OpenAI chat.completions için content listesi:
    - önce metin
    - ardından varsa birden fazla image_url
    """
    text_only = build_user_text(req)

    content: List[Dict[str, Any]] = [
        {"type": "text", "text": text_only},
    ]

    # Çok fazla görsel patlatmasın diye en fazla 3 tane alıyoruz
    for b64 in (req.screenshots_base64 or [])[:3]:
        if not b64:
            continue
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                },
            }
        )

    return content


# -------------------------------------------------
# Endpointler
# -------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    """
    Araç ilanı analizi:
    - text + (opsiyonel) screenshots_base64 listesi kullanır.
      (Flutter tarafında NormalAnalizScreen bunu gönderiyor.)
    """
    client = get_openai_client()

    system_prompt = (
        "Sen Oto Analiz uygulaması için bir uzman araç ilanı analiz asistanısın. "
        "Kullanıcının gönderdiği ilan metni ve varsa ekran görüntülerindeki bilgileri "
        "birlikte kullanarak, Türkiye ikinci el piyasasına göre detaylı bir analiz yap. "
        "Kronik sorun riskleri, muhtemel masraflar, aracın artı/eksi yönleri ve "
        "pazarlık tavsiyeleri ver. Cevabı kullanıcıya sade, anlaşılır ve maddeli şekilde yaz."
    )

    user_content = build_user_content(req)
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
            temperature=0.4,
        )

        result_text = resp.choices[0].message.content or "Boş çıktı"
        return AnalyzeResponse(ok=True, result=result_text)

    except HTTPException:
        # Yukarıda biz attıysak direkt yükselt
        raise
    except Exception as e:
        # Render loglarında görebil diye yaz
        print("OpenAI hata:", e)
        raise HTTPException(
            status_code=500,
            detail=f"OpenAI isteğinde hata oluştu: {str(e)}",
        )
