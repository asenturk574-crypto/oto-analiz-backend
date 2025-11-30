import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

# OpenAI client (OPENAI_API_KEY env'den gelecek)
client = OpenAI()

app = FastAPI()


class AnalyzeRequest(BaseModel):
    url: Optional[str] = ""
    usage: Optional[str] = ""
    description: Optional[str] = ""
    # Budget string olsun ki Flutter'dan gelen string sorun çıkarmasın
    budget: Optional[str] = None
    city: Optional[str] = ""
    note: Optional[str] = ""
    # "normal" veya "premium"
    mode: Optional[str] = "normal"


@app.get("/")
async def root():
    return {"status": "ok", "message": "Oto Analiz backend çalışıyor."}


@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    """
    Ana analiz endpoint'i.
    Flutter'dan gelen body buraya düşüyor.
    """
    result = await run_analysis(request)
    return result


async def run_analysis(request: AnalyzeRequest):
    if not request.description and not request.url:
        raise HTTPException(
            status_code=400,
            detail="Analiz için en az ilan açıklaması veya ilan linki (url) gerekir.",
        )

    parts = []

    # Kullanıcı profili
    if request.usage:
        parts.append(f"Kullanım amacı: {request.usage}")

    if request.budget is not None and request.budget != "":
        parts.append(f"Kullanıcının hedef bütçesi: {request.budget} TL civarı")

    if request.city:
        parts.append(f"Kullanıcının bulunduğu şehir: {request.city}")

    if request.note:
        parts.append(f"Kullanıcının ek notları / beklentileri:\n{request.note}")

    # Araç / ilan bilgisi
    if request.description:
        parts.append(f"İlan açıklaması:\n{request.description}")

    if request.url:
        parts.append(f"İlan linki (sadece referans): {request.url}")

    user_info_block = "\n\n".join(parts)

    # -------------------- PROMPT SEÇİMİ --------------------

    mode = (request.mode or "normal").lower()

    if mode == "premium":
        # UZUN, DETAYLI PREMIUM ANALİZ
        prompt = f"""
Sen Türkiye'de ikinci el araç piyasasını çok iyi bilen, yıllardır ekspertiz yapan
ve aynı zamanda kullanıcı dostu bir araç danışmanısın.

Aşağıda sana bir kullanıcının profili ve bir araç ilanının metinleri veriliyor.

Senin görevin, bu aracı KULLANICININ KRİTERLERİNE GÖRE analiz etmek ve
başlık başlık profesyonel ama anlaşılır bir rapor üretmek.

Her zaman KULLANICI PROFİLİ + ARAÇ PROFİLİ + TÜRKİYE PİYASASI
üzerinden mantıklı, tutarlı yorum yap.

Kullanıcı tüm alanları doldurmak zorunda değildir. Bir alan boşsa:
- "Bu bilgi verilmediği için genel Türkiye koşulları üzerinden yorum yapılmıştır." gibi not düş
- Asla analiz vermekten kaçınma, genel kullanım mantığına göre bilgi ver.

ÇIKTIMI MUTLAKA AŞAĞIDAKİ BAŞLIKLARLA VE SIRAYLA VER:

1) Araç Özeti
- Markayı, modeli, motor tipini, yılını, kilometreyi ilan metninden çıkarabildiğin kadar özetle.
- Aracın karakterini 2-3 cümleyle anlat (konfor, performans, aile kullanımı, segment, kimlere hitap ettiği).

2) Kullanıcı Profili & Uygunluk
- Kullanıcının kullanım amacı, bütçesi, şehri ve ek notlarına göre bir profil çıkar.
- Kullanıcının önceliklerini özetle (konfor, düşük masraf, performans, aile, park kolaylığı vb.).
- Bu araç bu profile ne kadar uyuyor? "Uygunluk: uygun / sınırda / zayıf" diye net bir ifade yaz.

3) Motor Analizi
- Motor tipine göre (dizel / benzin / LPG / hibrit) genel yorum yap.
- Turbo mu, atmosferik mi? Yokuşta ve tam dolu kullanımda performansı nasıl olur?
- 1.0–1.2 gibi küçük hacimli motor büyük bir kasada ise özellikle bunu belirt.
- Dizel + kısa mesafe kullanımda DPF/EGR riskini, LPG'de soğuk çalışmada ve uzun vadede yaşanabilecek sorunları açıkla.
- "Motor risk seviyesi: düşük / orta / yüksek" diye net bir cümle yaz.

4) Şanzıman Analizi
- İlan metninden şanzıman tipini (DSG/DCT, CVT, tork konvertörlü, robotize, manuel) anlamaya çalış.
- Her şanzıman tipi için:
  - DSG/DCT: performanslıdır ama dur-kalk trafikte kavrama ısınması ve pahalı arıza riskinden bahset.
  - CVT: şehir içi için konforlu ama yüksek yükte bağırma ve kayış/kasnak maliyetinden bahset.
  - Tork konvertörlü: dayanıklılığı, ağır kasada avantajını, arıza olursa revizyon maliyetini anlat.
  - Robotize: düşük hızda vuruntu, kavrama aşınması ama nispeten düşük tamir maliyetinden bahset.
  - Manuel: masraf açısından avantajlı ama yoğun trafikte yorucu olduğunu anlat.
- Kullanıcının şehir içi / uzun yol kullanımına göre bu şanzımanın uygunluğunu yorumla.
- "Şanzıman risk seviyesi: düşük / orta / yüksek" diye yaz.

5) Araç Boyutu & Kullanım Ortamı
- Araç B/C/D segment mi, sedan mı, hatchback mi, SUV mu tahmin etmeye çalış.
- Şehir içi dar sokak, park sorunu gibi durumlarda büyük gövdeli araçların dezavantajlarını anlat.
- Uzun yol ve aile için geniş sedan/SUV'in avantajlarını, şehir içi parkta dezavantajlarını belirt.

6) Yakıt Uygunluğu
- Kullanıcının yakıt tercihi ile aracın yakıt tipini karşılaştır.
- Dizel + uzun yol için avantajları, dizel + kısa mesafe için riskleri açıkla.
- Benzinli için yakıt tüketimi / sessizlik, LPG için masraf avantajı / bagaj kaybı ve ayar sorunlarından bahset.
- "Yakıt uygunluğu: uygun / sınırda / zayıf" diye net bir ifade yaz.

7) Kronik Sorunlar & Risk Listesi
- Bu araç tipi/motor/şanzıman için bilinen kronik sorunları 3–6 maddelik bir listede yaz.
- Her maddenin sonuna parantez içinde risk seviyesi yaz: (risk: düşük), (risk: orta), (risk: yüksek).

8) Masraf Analizi (Kısa & Orta Vade)
- Yakın vadede (bakım, lastik, fren, ufak tamirler) çıkabilecek muhtemel masraf seviyesini yorumla.
- Orta vadede (triger, debriyaj, şanzıman yağı, turbo bakımı, enjektör) muhtemel masraf seviyesini anlat.
- Rakam verme, "düşük / orta / yüksek" seviye kullan.
- "Genel masraf seviyesi: düşük / orta / yüksek" diye net bir cümle yaz.

9) Sigorta, Kasko ve Parça Bulunabilirliği
- Bu segment ve yaşta bir araç için kasko ve sigorta maliyet seviyesini (düşük/orta/yüksek) yorumla.
- Parça bulunabilirliği: kolay / orta / zor şeklinde değerlendir.
- Özel servis ve yan sanayi parça ile kullanılabilirlik hakkında genel bir yorum yap.

10) Piyasa & Satılabilirlik
- Türkiye ikinci el piyasasında bu modelin tutulup tutulmadığını yorumla.
- Motor/şanzıman kombinasyonunun alıcı kitlesi geniş mi dar mı, bunu belirt.
- Piyasasını "Piyasa: hızlı / normal / yavaş" olarak değerlendir.
- Satılabilirliği "Satılabilirlik: kolay / ortalama / zor" diye yaz.

11) Kullanıcıya Özel Uyarılar
- Özellikle KULLANICI PROFİLİ ile ARAÇ PROFİLİNİN çeliştiği noktaları maddeler halinde yaz.
- Örn: kısa mesafe + dizel, küçük motor + ağır kasa, DSG + yoğun trafik, büyük kasa + dar sokaklar, bütçe sınırı vs.
- En az 3, en fazla 7 madde yaz.

12) Fiyat & Pazarlık Yorumu
- Verilen bütçe bilgisine ve araç tipine göre fiyatın "uygun / normal / yüksek" olup olmadığını yorumla.
- Net rakam söyleme, ama pazarlık payı hakkında genel yorum yap.
- "Bu fiyata daha düşük km/daha temiz alternatif bulunabilir mi?" sorusuna kısaca değin.

13) Alternatif Araç Önerileri
- Kullanıcının profilini ve bütçesini dikkate alarak 2–3 alternatif model öner.
- Her araç için çok kısa sebep yaz (daha düşük masraf, şehir içi uygunluğu, aile kullanımı, daha sorunsuz şanzıman vb.).
- Marka fanlığı yapma, objektif kal.

14) Puanlama (10 Üzerinden)
- Alt başlıklarda 10 üzerinden puan ver:
  - Kullanım amacına uygunluk
  - Motor-masraf dengesi
  - Şanzıman güvenilirliği
  - Yakıt-maliyet dengesi
  - Aile/şehir içi uygunluğu
  - İkinci el piyasası
- En sonda "Genel skor: X / 10" yaz.

15) Sonuç & Karar (TEK NET CÜMLE)
- En sonda ayrı bir satırda şu formatta net bir karar yaz:
  - "Karar: Ekspertiz temizse alınabilir."
  - "Karar: Ancak ciddi pazarlıkla değerlendirilebilir."
  - "Karar: Bu kilometrede bu fiyata çok mantıklı değil."
  - "Karar: Bütçeyi biraz artırarak daha temiz seçenek bakmak daha mantıklı olur."

KULLANICI VE ARAÇ BİLGİLERİ:

{user_info_block}
"""
        model_name = "gpt-4.1"

    else:
        # NORMAL (ÜCRETSİZ) ANALİZ – KISA AMA KİŞİYE ÖZEL
        prompt = f"""
Sen Türkiye'de ikinci el araç piyasasını iyi bilen bir araç danışmanısın.
Aşağıdaki bilgiler bir kullanıcının profili ve bir araç ilanı hakkındadır.

NORMAL analiz modundasın.
Bu modda PREMIUM kadar detay VERME.
Kısa, net ve kişiye özel yorum yap.

Analizi şu başlıklarla ve KISA şekilde yaz:

1) Araç Özeti (1–2 cümle)
- Markayı, modeli, motoru ve genel durumu ilan metninden çıkarabildiğin kadar özetle.

2) Sizin Kullanım Amacınıza Uygunluk
- Kullanıcının kullanım amacı, şehri ve notlarına göre bu aracın ne kadar uygun olduğunu 1 cümlede açıkla.
- Sonuna parantez içinde "uygun / sınırda / pek uygun değil" yaz.

3) Motor & Yakıt Değerlendirmesi
- Motor tipi (dizel/benzin/LPG/hibrid) ve kullanıcı beklentisine göre yakıt-masraf dengesini 1–2 cümlede özetle.
- Sonunda "risk seviyesi: düşük / orta / yüksek" ekle.

4) Şanzıman Yorumu
- İlan metninden şanzıman tipini tahmin etmeye çalış (otomatik/manuel, DSG/CVT/Tork vb. mümkünse).
- Tek cümlede şehir içi kullanım açısından konfor ve güvenilirlik yorumu yap.
- Sonuna "güvenilirlik: düşük / orta / yüksek" yaz.

5) Artılar (Sizin kullanımınıza göre)
- Kullanıcının kullanım amacı, şehir ve beklentilerine göre bu araç için 2 madde artı yaz.
- Maddeler kısa olsun.

6) Eksiler (Sizin kullanımınıza göre)
- Aynı şekilde, bu kullanıcı için 2 madde eksi yaz.
- Maddeler kısa olsun.

7) Fiyat Yorumu
- Verilen bütçeyi ve araç tipini dikkate alarak fiyatın piyasaya göre "uygun / normal / biraz yüksek" olduğunu 1 cümlede yorumla.

8) Sonuç (tek cümle)
- "Alınabilir", "Pazarlıkla değerlendirilebilir" veya "Daha temiz alternatiflere bakmak daha mantıklı" formatında net ve kısa bir karar yaz.

Gereksiz uzun anlatım yapma, kullanıcıyı yormadan net konuş.

KULLANICI VE ARAÇ BİLGİLERİ:

{user_info_block}
"""
        model_name = "gpt-4.1-mini"

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Sen uzman bir ikinci el araç ekspertiz ve fiyat değerlendirme danışmanısın. "
                        "Kullanıcıya net, dürüst, anlaşılır ve kişiye özel yorum yap."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
        )

        analysis_text = completion.choices[0].message.content

        return {
            "mode": mode,
            "analysis": analysis_text,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analiz sırasında hata oluştu: {e}",
        )
