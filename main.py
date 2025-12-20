import os
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from openai import OpenAI


app = FastAPI(title="oto-analiz-backend", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY missing on server. Render Environment'a ekle.",
        )
    return OpenAI(api_key=api_key)


def build_user_text(req: AnalyzeRequest) -> str:
    """Metin alanlarını tek bir stringe çevirir."""
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

    return "\n\n".join(parts)


def build_user_content(req: AnalyzeRequest) -> List[Dict[str, Any]]:
    """
    OpenAI chat.completions için content listesi:
    - önce text
    - sonra varsa birden fazla input_image
    """
    text = build_user_text(req)
    content: List[Dict[str, Any]] = [
        {"type": "text", "text": text},
    ]

    for b64 in (req.screenshots_base64 or []):
        if not b64:
            continue
        content.append(
            {
                "type": "input_image",
                "image_url": {
                    # frontend base64 (jpeg/png) gönderiyor
                    "url": f"data:image/jpeg;base64,{b64}",
                },
            }
        )

    return content


@app.get("/")
def health():
    return {"status": "ok"}


@app.get("/health")
def health2():
    return {"ok": True}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    """
    Araç ilanı analizi:
    - text + (opsiyonel) screenshots_base64 listesi kullanır.
    """
    client = get_openai_client()

    system_prompt = (
        "Sen Oto Analiz uygulaması için bir uzman araç ilanı analiz asistanısın. "
        "Kullanıcının gönderdiği ilan metni ve varsa ekran görüntüsündeki bilgileri kullanarak, "
        "Türkiye ikinci el piyasasına göre detaylı bir analiz yap. "
        "Kronik sorun riskleri, muhtemel masraflar, artı/eksi yönler ve pazarlık tavsiyesi ver. "
        "Metni kullanıcıya sade ve anlaşılır biçimde Türkçe yaz."
    )

    user_content = build_user_content(req)

    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.4,
        )

        result_text = resp.choices[0].message.content or "Boş çıktı"

        return AnalyzeResponse(ok=True, result=result_text)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI request failed: {str(e)}")
