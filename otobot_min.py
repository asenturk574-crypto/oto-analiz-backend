"""
Mini OtoBot (V1) — 2 özellik:
1) Araç Öner (ilan yok, fiyat yok)
2) Alım Rehberi (satıcı/ekspertiz/test sürüşü checklist)

Bu modül FastAPI router olarak gelir. Mevcut main.py dosyana sadece:
    from otobot_min import router as otobot_router
    app.include_router(otobot_router)
eklemen yeterli.

ENV:
- OPENAI_API_KEY (opsiyonel — yoksa kural tabanlı fallback çalışır)
- OPENAI_MODEL_OTOBOT (opsiyonel, default: gpt-4.1-mini)
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Literal, Optional, Tuple

from fastapi import APIRouter
from pydantic import BaseModel, Field

try:
    from openai import OpenAI  # openai>=1.x
except Exception:
    OpenAI = None  # type: ignore


# ===============================
# OpenAI client
# ===============================
def _get_client() -> Optional["OpenAI"]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or OpenAI is None:
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception:
        return None


OPENAI_MODEL_OTOBOT = os.getenv("OPENAI_MODEL_OTOBOT", "gpt-4.1-mini")


# ===============================
# Router
# ===============================
router = APIRouter(prefix="", tags=["otobot"])


# ===============================
# Models
# ===============================
class OtoBotState(BaseModel):
    """
    Sunucu state tutmaz. İstemci bu state'i saklar ve sonraki isteklerde geri yollar.
    """
    step: int = 0
    answers: Dict[str, Any] = Field(default_factory=dict)


class OtoBotRequest(BaseModel):
    mode: Literal["recommend", "guide"] = "recommend"
    message: Optional[str] = None

    # recommend akışı için state
    state: Optional[OtoBotState] = None

    # Kullanıcı "5 öneriden hangisini seçtin?" derse client bunu buraya koyabilir (1-5)
    selected_index: Optional[int] = Field(default=None, ge=1, le=5)

    # Alım rehberi, seçime göre (tip hedefleme) özelleştirmede yardımcı
    selected_style: Optional[str] = None


class Card(BaseModel):
    title: str
    bullets: List[str]


class Recommendation(BaseModel):
    label: str
    type: str
    why: List[str]
    watch_out: str


class OtoBotResponse(BaseModel):
    type: Literal["question", "recommendations", "guide", "info"] = "info"
    message: str
    state: Optional[OtoBotState] = None
    recommendations: Optional[List[Recommendation]] = None
    cards: Optional[List[Card]] = None


# ===============================
# Conversation schema (Recommend)
# ===============================
QUESTIONS: List[Tuple[str, str]] = [
    ("budget_band", "Bütçen yaklaşık hangi aralıkta? (örn: 700–900K)"),
    ("city", "Hangi şehirde kullanacaksın?"),
    ("yearly_km", "Yıllık yaklaşık kaç km yaparsın? (örn: 8.000 / 15.000 / 30.000+)"),
    ("usage", "Kullanımın daha çok ne? (Şehir içi / Uzun yol / Karışık)"),
    ("auto_required", "Otomatik şart mı? (Evet/Hayır)"),
    ("fuel_pref", "Yakıt tercihin var mı? (Benzin / Dizel / Hibrit) — LPG düşünür müsün?"),
    ("priority", "Senin için en önemli şey hangisi? (Ekonomi / Sorunsuzluk / Konfor / Performans / Genişlik)"),
]


# ===============================
# Helpers
# ===============================
def _normalize_yes_no(s: str) -> Optional[bool]:
    s2 = (s or "").strip().lower()
    if not s2:
        return None
    if any(x in s2 for x in ["evet", "e", "yes", "y", "olmazsa olmaz", "şart"]):
        return True
    if any(x in s2 for x in ["hayır", "h", "no", "n", "farketmez", "fark etmez", "olmasa da olur"]):
        return False
    return None


def _to_int_maybe(s: Any) -> Optional[int]:
    try:
        if isinstance(s, (int, float)):
            return int(s)
        s2 = str(s)
        digits = re.sub(r"[^\d]", "", s2)
        if not digits:
            return None
        return int(digits)
    except Exception:
        return None


def _safe_trim(s: str, n: int = 700) -> str:
    s = (s or "").strip()
    if len(s) <= n:
        return s
    return s[: n - 3] + "..."


def _json_loads_loose(text: str) -> Optional[dict]:
    """
    Model bazen JSON dışında bir şey ekleyebilir.
    İlk { ... } bloğunu yakalayıp parse etmeye çalışır.
    """
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


# ===============================
# Fallback logic (no OpenAI)
# ===============================
def _fallback_recommendations(a: Dict[str, Any]) -> List[Recommendation]:
    usage = str(a.get("usage", "")).lower()
    city = str(a.get("city", "")).lower()
    auto_req = a.get("auto_required")
    fuel = str(a.get("fuel_pref", "")).lower()
    prio = str(a.get("priority", "")).lower()
    km = _to_int_maybe(a.get("yearly_km")) or 0

    traffic_heavy = any(x in city for x in ["istanbul", "ankara", "izmir", "bursa"]) or "şehir" in usage
    wants_auto = auto_req is True or (isinstance(auto_req, str) and "evet" in auto_req.lower())

    # Basit tip önerisi: ilan/fiyat yok, sadece segment/şanzıman/motor tipi
    recs: List[Recommendation] = []

    # #1 dengeli
    recs.append(Recommendation(
        label="En dengeli",
        type=("B/C segment • Hatch/Sedan • " +
              ("Otomatik " if wants_auto else "") +
              ("tam hibrit" if "hib" in fuel else "benzin")),
        why=[
            "Günlük kullanımda pratik, bakım/işletme tarafı genelde daha yönetilebilir.",
            "Şehir içi ağırlıkta sürüş konforu ve kullanım kolaylığı daha yüksek.",
        ],
        watch_out="Donanım/şanzıman tipi çok değişir; sürüşte sarsıntı ve gecikme var mı bak.",
    ))

    # #2 sorunsuz odaklı
    recs.append(Recommendation(
        label="Sorunsuzluk odaklı",
        type=("B segment • Atmosferik benzin • " + ("Otomatik (tork konvertör/klasik)" if wants_auto else "Manuel")),
        why=[
            "Basit mekanik yapı uzun vadede sürpriz riskini azaltma eğilimindedir.",
            "Parça/servis erişimi genelde daha rahat olur.",
        ],
        watch_out="Yaşı yüksek araçta bakım geçmişi şart; yağ kaçakları/soğutma sistemi kontrol.",
    ))

    # #3 ekonomik
    recs.append(Recommendation(
        label="Ekonomi odaklı",
        type=("B segment • " + ("tam hibrit" if traffic_heavy else "benzin") + " • Otomatik"),
        why=[
            "Şehir içi yoğun trafikte tüketim avantajı sunabilir (özellikle hibritte).",
            "Küçük hacim ve düşük ağırlık işletme maliyetlerini düşürmeye yardımcı olur.",
        ],
        watch_out="Batarya/hibrid sistem sağlık kontrolü (genel kontrol) yaptır.",
    ))

    # #4 aile/konfor
    recs.append(Recommendation(
        label="Aile/konfor odaklı",
        type="C segment • Sedan/SUV • Otomatik",
        why=[
            "Daha geniş iç hacim ve bagaj, aile kullanımını rahatlatır.",
            "Uzun yolda stabilite ve konfor artar.",
        ],
        watch_out="SUV tercihinde lastik/fren maliyeti daha yüksek olabilir; sürüşte ses ve titreşime bak.",
    ))

    # #5 keyif/perf (talebe göre)
    recs.append(Recommendation(
        label="Keyif/performans (isteğe bağlı)",
        type=("C segment • Turbo benzin • " + ("Otomatik" if wants_auto else "Manuel")),
        why=[
            "Ara hızlanmalar ve sürüş keyfi daha yüksek olur.",
            "Uzun yolda sollamalarda rahatlık sağlar.",
        ],
        watch_out="Turbo/soğutma ve düzenli bakım kritik; agresif kullanım izleri varsa uzak dur.",
    ))

    # Prio bazlı küçük düzenleme
    if "konfor" in prio:
        recs[0].type = recs[3].type
        recs[0].why = ["Konfor ve yol stabilitesi önceliğine daha iyi uyar.", "Aile/şehir dışı kullanımda daha rahat hissettirir."]
    if "perform" in prio:
        recs[0].type = recs[4].type
        recs[0].why = ["Önceliğin performanssa daha tatmin edici sürüş verir.", "Günlük kullanımda da dengeli kalabilir."]
    if "ekonom" in prio:
        recs[0].type = recs[2].type
        recs[0].why = ["Önceliğin ekonomi ise günlük gideri daha iyi yönetirsin.", "Şehir içi kullanımda avantaj sağlar."]
    if km >= 25000 and "uzun" in usage:
        # Uzun yol + yüksek km: (ilan yok) diesel önerisi yerine genel ifade
        recs[3].why.append("Yıllık km yüksekse konfor + stabilite daha da önem kazanır.")
        recs[3].watch_out = "Bakım aralıklarını kaçırma; uzun km yapan araçta servis kaydı en kritik şey."

    return recs


def _guide_cards(selected_style: Optional[str] = None) -> List[Card]:
    """
    Alım rehberi: 3 kart. (İlan yok. Model söyleme yok. Genel ama hedefli.)
    selected_style: kullanıcı seçimi (örn. 'B segment otomatik', 'SUV', vs.) — sadece dil tonunu hedefler.
    """
    style = (selected_style or "").strip()
    prefix = f"({style}) " if style else ""

    card1 = Card(
        title=f"{prefix}Satıcıyla konuşma: 8 soru",
        bullets=[
            "Hasar geçmişi detayını (kalem kalem) paylaşır mısınız?",
            "Bakım kayıtları/fatura var mı? En son bakım ne zaman yapıldı?",
            "Aracı kaç yıldır kullanıyorsunuz, kaçıncı sahipsiniz?",
            "Ekspertizi nerede yapalım? (kabul ediyor musunuz)",
            "Soğuk çalıştırma mümkün mü? (ilk marş)",
            "Büyük işlem oldu mu? (motor/şanzıman/kaporta)",
            "Lastiklerin durumu ve yaşı nedir?",
            "Muayene, 2 anahtar, ruhsat/rehin/çekme vb. bir durum var mı?",
        ],
    )

    card2 = Card(
        title=f"{prefix}Ekspertizde mutlaka baktır: 10 madde",
        bullets=[
            "Boya/değişen parçaların kalem kalem tespiti",
            "Şasi/podye/direk gibi kritik bölgeler",
            "Motor: yağ kaçakları + soğutma sistemi kontrolü",
            "Şanzıman: kaçak + geçiş kontrolü (varsa)",
            "Alt takım: salıncak/rot/aks, burçlar",
            "Fren: disk/balata + ABS kontrolü",
            "Direksiyon sistemi boşluk/ses kontrolü",
            "Klima performansı",
            "Elektronik arıza taraması (OBD)",
            "Akü/şarj sistemi",
        ],
    )

    card3 = Card(
        title=f"{prefix}Test sürüşü: 6 kontrol + son adım",
        bullets=[
            "İlk 5 dk: ses/titreşim/çekiş anormalliği var mı?",
            "Direksiyon düz giderken sağa-sola çekiyor mu?",
            "Frenleme: titreme/sağa çekme var mı?",
            "Kasiste/vibrasyonda vuruntu veya boşluk sesi var mı?",
            "10–15 dk sonra: hararet normal mi? fan düzenli çalışıyor mu?",
            "Park manevrası: direksiyonda tıkırtı/sertlik var mı?",
            "Son adım: Temiz çıksa bile masrafları (bakım/lastik/kozmetik) listeleyip pazarlığa öyle gir.",
        ],
    )
    return [card1, card2, card3]


# ===============================
# OpenAI path (optional)
# ===============================
SYSTEM_PROMPT = """Sen Oto Analiz uygulamasının Mini OtoBot'sun.
Sadece iki şey yaparsın:
1) Kullanıcı profiline göre ilan olmadan GENEL araç tipi önerisi (5 öneri: dengeli/sorunsuz/ekonomik/aile/keyif).
2) Araç alım rehberi (satıcı soruları + ekspertiz + test sürüşü checklist).

Kurallar:
- Fiyat söyleme.
- Skor/puan üretme.
- Yıllık maliyet bandı hesaplama.
- Araç-özel kronik listeleme.
- 'Alınır/Alınmaz' kesin hüküm verme.
- Cevaplar kısa, net, madde madde olsun.
- Öneriler 'segment + kasa tipi + motor/vites tipi' formatında, 2 neden + 1 dikkat maddesiyle gelsin.

ÇIKTIYI MUTLAKA JSON döndür.
"""

def _openai_recommendations(answers: Dict[str, Any]) -> Optional[List[Recommendation]]:
    client = _get_client()
    if client is None:
        return None

    payload = {
        "answers": answers,
        "instruction": "Kullanıcı profiline göre 5 genel öneri üret. Her öneri: label, type, why(list), watch_out. JSON dön."
    }

    try:
        # Use responses API (OpenAI python SDK 1.x)
        resp = client.responses.create(
            model=OPENAI_MODEL_OTOBOT,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
        text = ""
        for item in resp.output:
            if item.type == "message":
                for c in item.content:
                    if c.type == "output_text":
                        text += c.text

        data = _json_loads_loose(text)
        if not isinstance(data, dict):
            return None

        recs = data.get("recommendations")
        if not isinstance(recs, list) or len(recs) < 3:
            return None

        out: List[Recommendation] = []
        for r in recs[:5]:
            if not isinstance(r, dict):
                continue
            out.append(Recommendation(
                label=str(r.get("label", "")).strip()[:40] or "Öneri",
                type=str(r.get("type", "")).strip()[:140] or "Tip",
                why=[_safe_trim(str(x), 120) for x in (r.get("why") or [])][:3] or ["Profiline uygun genel bir seçenek."],
                watch_out=_safe_trim(str(r.get("watch_out", "") or "Alım rehberine göre kontrol et."), 140),
            ))
        if len(out) >= 3:
            # 5'e tamamla (gerekirse fallback ile)
            while len(out) < 5:
                out.append(_fallback_recommendations(answers)[len(out)])
            return out[:5]
        return None
    except Exception:
        return None


# ===============================
# Endpoints
# ===============================
@router.post("/otobot", response_model=OtoBotResponse)
def otobot(req: OtoBotRequest) -> OtoBotResponse:
    """
    Tek endpoint:
    - mode=recommend: soru akışı -> 5 öneri
    - mode=guide: 3 kart alım rehberi
    """
    if req.mode == "guide":
        cards = _guide_cards(req.selected_style)
        return OtoBotResponse(
            type="guide",
            message="Alım rehberi hazır. Aşağıdaki kontrol listelerini adım adım uygula.",
            cards=cards,
        )

    # recommend mode
    state = req.state or OtoBotState(step=0, answers={})
    msg = (req.message or "").strip()

    # Eğer kullanıcı seçim yaptıysa: rehbere yönlendir
    if req.selected_index is not None:
        idx = int(req.selected_index)
        style = req.selected_style or f"Öneri #{idx}"
        cards = _guide_cards(style)
        return OtoBotResponse(
            type="guide",
            message=f"Tamam. {style} için alım rehberini hazırladım.",
            cards=cards,
            state=None,
        )

    # step soru
    if state.step < len(QUESTIONS):
        key, question = QUESTIONS[state.step]

        # Kullanıcı bu adımın cevabını yazdıysa kaydet ve bir sonraki soruya geç
        if msg:
            # normalize some fields
            if key == "auto_required":
                yn = _normalize_yes_no(msg)
                state.answers[key] = yn if yn is not None else msg
            else:
                state.answers[key] = msg

            state.step += 1

        # Eğer hala soru kalmışsa sor
        if state.step < len(QUESTIONS):
            _, q = QUESTIONS[state.step]
            return OtoBotResponse(
                type="question",
                message=q,
                state=state,
            )

    # Sorular bitti -> öneri üret
    answers = dict(state.answers)

    recs = _openai_recommendations(answers) or _fallback_recommendations(answers)

    # response message
    end_msg = (
        "Tamam. Profiline göre 5 mantıklı GENEL seçenek çıkardım (ilan/fiyat yok).\n"
        "Beğendiğin numarayı yaz (1–5). İstersen seçtiğine göre alım rehberi de çıkarırım."
    )

    return OtoBotResponse(
        type="recommendations",
        message=end_msg,
        recommendations=recs,
        state=None,
    )
