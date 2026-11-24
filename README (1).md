# IŞIKTA KAL

Elektrik şirketlerinin fatura, ödeme ve tüketim verisini kullanarak gizli enerji yoksulluğunu (faturasını ödediği halde tüketimini kısıtlayan haneleri) ve kesinti riskini erken tespit etmeye yönelik bir kavram kanıtı çalışması.

Yenilikçi Fikirler Kampı 2026 başvurusu için hazırlanmıştır.
Sude Çatalkaya, İzmir Bakırçay Üniversitesi, Bilgisayar Mühendisliği

Prototip: https://12345sude.github.io/isiktakal/app/

## Dosyalar

- `isiktakal.py`: sentetik veri üretimi, Kısıt İndeksi, kesinti riski modeli
- `gercek_veri.py`: Kısıt İndeksi'nin Low Carbon London gerçek verisiyle testi (veriyi kendisi indirir)
- `gercek_grafikler.py`: gerçek veri şekilleri
- `grafikler.py`: rapordaki şekiller
- `uygulama_olustur.py`: model çıktılarını operatör paneline aktarır
- `app/sablon.html`: operatör paneli
- `app/index.html`: veriler eklenmiş, tek dosyalık çalışan uygulama (tarayıcıda açılabilir)

## Çalıştırma

```
pip install -r requirements.txt
python isiktakal.py
python grafikler.py
python gercek_veri.py
python gercek_grafikler.py
python uygulama_olustur.py
```

Son komut `app/index.html` dosyasını yeniden oluşturur. Dosya doğrudan tarayıcıda açılabilir veya GitHub Pages ile yayınlanabilir.

## Sonuçlar: sentetik veri

| | Yalnızca ödeme | Ödeme + Kısıt İndeksi |
|---|---|---|
| PR-AUC | 0,64 | 0,68 |
| En riskli %5'te yakalanan kesinti | %82 | %85 |
| Kısıtlayıp borca kayan hanelerde uyarı süresi | 105 gün | 140 gün |

Kısıt tespiti: kesinlik 0,75, duyarlılık 0,86.

## Sonuçlar: gerçek veri (Low Carbon London)

3.196 Londra hanesi, öğrenme dönemi Temmuz 2012 – Haziran 2013, test dönemi Temmuz – Aralık 2013. Dinamik tarife deneyindeki haneler hariç.

- Hava koşuluna göre beklenen tüketim tahmininin ortanca hatası: %12,8
- İlk kural 146 haneyi işaretledi, ancak işaretlenme varlıklı bölgelerde daha sıktı (p = 0,005). Bunların 76'sının tüketimi yazın da düşüktü (evden uzak kalma, taşınma vb.).
- Yaz kontrolü eklendiğinde 70 hane işaretlendi ve ACORN grupları arasındaki fark kayboldu (p = 0,22).
- Kış–yaz göreli indeks farkı dezavantajlı bölgelerde biraz daha düşük (p = 0,028).

Ödeme verisi olmadığı için kesinti riski modeli gerçek veride test edilmedi.

Veri kaynağı: UK Power Networks, "SmartMeter Energy Consumption Data in London Households", London Datastore, CC BY 4.0. Hane-ay toplamları github.com/Nasheor/lcl-dtou-causal deposundan, hava ve ACORN bilgisi github.com/Iavalos/Forecasting-electricity-consumption-in-London deposundan alınmaktadır.

## Not

Sentetik veride tüketim kısıtlamasının borçtan önce geldiği varsayımı simülasyona eklenmiştir ve gerçek veriyle test edilmesi gerekir. Operatör paneli ödeme verisi gerektirdiği için sentetik veriyle çalışır.
