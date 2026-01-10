KURULUM:
1) Bu zip içindeki main.py ve otobot_min.py dosyalarını backend repo köküne koy.
2) Render deploy.

NOTLAR:
- /otobot endpoint otobot_min.py içindeki router'dan gelir.
- Eski otobot endpoint artık /otobot_legacy.
- Premium analiz (build_premium_template) kart/başlık yapısına göre güncellendi.
- Response: 'cards' -> [{title, lines, content}] ve 'result' -> kartların metinleştirilmiş hali.
