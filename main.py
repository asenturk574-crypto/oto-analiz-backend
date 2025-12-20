import os
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from openai import OpenAI


# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title="oto-analiz-backend", version="1.0.0")

# CORS (Flutter rahat etsin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # prod'da istersen domain bazlı kısıtlarız
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------
# Models
# -------------------------
class AnalyzeRequest(BaseModel):
    # Flutter'dan en az birini gönderebilirsin
    text: Optional[str] = None
    listing_title: Optional[str] = None
    description: Optional[str] = None

    # ekstra parametreler (opsiyonel)
    extras: Optional[Dict[str, Any]] = None


class AnalyzeResponse(BaseModel):
    ok: bool
    result: str


# -------------------------
# Helpers
# -------------------------
def get_openai_client() -> OpenAI:
    """
    Render'da .env yok; Environment Variables'dan OPENAI_API_KEY gelmeli.
    Server crash olmasın diye sadece endpoint çağrılınca kontrol ediyoruz.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY missing on server. Render -> Environment bölümüne ekle.",
        )
    return OpenAI(api_key=api_key)


def build_user_input(req: AnalyzeRequest) -> str:
    parts = []
    if req.text:
        parts.append(f"USER_TEXT:\n{req.text}")
    if req.listing_title:
        parts.append(f"LISTING_TITLE:\n{req.listing_title}")
    if req.description:
        parts.append(f"DESCRIPTION:\n{req.description}")
    if req.extras:
        parts.append(f"EXTRAS:\n{req.extras}")

    if not parts:
        # 422 yerine daha anlaşılır hata
        raise HTTPException(
            status_code=400,
            detail="Boş istek. En azından 'text' veya 'listing_title/description' gönder.",
        )
    return "\n\n".join(parts)


# -------------------------
# Routes
# -------------------------
@app.get("/")
def health():
    return {"status": "ok"}


@app.get("/health")
def health2():
    return {"ok": True}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    """
    Basit analiz endpoint'i.
    Flutter JSON örneği:
    {
      "text": "2016 Passat 1.6 TDI DSG 180.000 km, fiyat 950.000 TL"
    }
    """
    user_input = build_user_input(req)

    client = get_openai_client()

    # Model adını burada merkezi yönetiyoruz
    # Not: Kullandığın SDK ve hesabına göre uygun model adı değişebilir.
    # Eğer hata alırsan logs'u at, birlikte doğru modele çeviririz.
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    system_prompt = (
        "Sen Oto Analiz uygulaması için araç ilanı analizi yapan bir asistansın. "
        "Kullanıcıya net, kısa, maddeli ve Türkiye piyasasına uygun yorum yap. "
        "Varsa riskleri ve dikkat edilmesi gerekenleri belirt. "
        "Tahmini masraf kalemleri ve pazarlık önerisi ekle."
    )

    try:
        # openai python sdk (new)
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
            temperature=0.4,
        )

        result_text = resp.choices[0].message.content or "Boş çıktı"

        return AnalyzeResponse(ok=True, result=result_text)

    except HTTPException:
        raise
    except Exception as e:
        # Render logs'ta da görürsün
        raise HTTPException(status_code=500, detail=f"OpenAI request failed: {str(e)}")


# İstersen buraya ileride:
# - /analyze-from-image (SS analiz)
# - /compare (karşılaştırma)
# - /pricing (fiyat tahmini)
# ekleriz.
