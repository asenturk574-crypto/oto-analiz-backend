Oto Analiz – Veri Paketi v2 (227 model profili)

Bu paket, premium analizde “zeka” hissini artırmak için:
- Daha çok model/segment eşleşmesi (227 TR odaklı model profili)
- Tag-bazlı risk kalıpları (DSG/DCT/CVT, turbo, dizel/DPF, LPG, hibrit/EV, AWD, PureTech/TSI/TDI vb.)
- Segment bazlı ekspertiz checklist’leri
sağlar.

1) Backend’e kopyalama
   backend/data/ klasörüne şu dosyaları koy:
   - anchors_tr_popular_v2_227.json
   - vehicle_profiles_v2_227.json
   - risk_patterns_by_tag_v1.json
   - big_maintenance_watchlist_by_tag_v1.json
   - inspection_checklists_by_segment_v1.json

2) main.py içinde loader’ları güncelle
   Eski:
     anchors_tr_popular_96.json
     vehicle_profiles_96_v1.json
   Yeni:
     anchors_tr_popular_v2_227.json
     vehicle_profiles_v2_227.json

3) Önerilen mini iyileştirme (çok etkili)
   build_enriched_context içinde:
   - guessed_tags = vp.tags + _guess_tags(req) şeklinde birleştir.
   Böylece “model profili” özel tag’leri (ör. tsi, dsg, multijet) çalışır.

Not:
- Bu paket “heuristik”tir: amaç tutarlı, piyasa gerçekliğine yakın ve güvenli tahmin üretmek.
- En doğru sonuç: ilan açıklaması + servis geçmişi + ekspertizle birlikte.
