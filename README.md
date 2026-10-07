# WashMom Vision

Kıyafet fotoğrafından kumaş sınıfı (EfficientNetV2B0) ve renk grubu (OpenCV) çıkarıp açıklanabilir kurallarla yıkama grubu öneren görüntü işleme projesi.

> Ticari amaç gütmeyen bir **öğrenme projesidir**. Lisans: [CC BY-NC 4.0](LICENSE) (telif ve veri seti notu: [NOTICE](NOTICE)). Ticari kullanım yasaktır (ayrıntılar: [Lisans](#lisans)).

## Sonuçlar (test seti, 3.231 garment crop)

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---|---|---|
| Custom CNN | 0,771 | 0,644 | 0,449 | 0,449 |
| EfficientNetV2B0 (frozen) | 0,827 | 0,704 | 0,748 | 0,718 |
| **EfficientNetV2B0 (fine-tuned)** | **0,831** | **0,740** | **0,736** | **0,730** |

- 6 kumaş sınıfı: cotton, denim, chiffon, knitted, leather, furry (ciddi sınıf dengesizliği → ana metrik Macro F1).
- Ürün numarası bazlı split: train / validation / test arasında ortak ürün yok (data leakage kontrolü).
- Confidence eşiği 0,55: kararların %90,6'sı otomatik (%86,6 doğru), %9,4'ü kullanıcıya soruluyor.

Örnek çıktı:

```json
{"fabric": "denim", "fabric_confidence": 0.98, "color_group": "dark",
 "washing_group": "DARK_HEAVY", "needs_label_check": false}
```

## Notebook

- Tüm proje tek notebook'ta: **`WashMom_Vision.ipynb`** (bölümlere ayrılmış, baştan sona tekrar çalıştırılabilir; kayıtlı model varsa yeniden eğitmez).
- **Bu repodaki notebook'ta veri seti görüntüleri yoktur** (örnek crop'lar, maskeler, hata galerisi, demo). Veri setinin lisansı görüntülerin yayınlanmasına izin vermiyor. Kod, tablolar, sayılar ve grafikler aynıdır.
- Kod Google Colab GPU'sunda çalıştırıldı. Notebook veriyi **kendi** Google Drive'ınızda aşağıdaki klasör yapısında bekler; bu klasör paylaşılmaz, veri setini resmi sayfadan kendiniz indirip buraya koymanız gerekir (bu repoda yalnızca demo modeli `models/` altında):

```
MyDrive/WashMom_Vision/
├── data_raw/     # labels.zip, segm.zip, images.zip (orijinal dataset, dokunulmaz)
├── processed/    # mask_stats.csv, crops.zip, manifest.csv, manifest_final.csv
├── models/       # eğitilmiş modeller (.keras) ve eğitim geçmişleri
└── outputs/      # grafikler, karşılaştırma tabloları, test sonuçları
```

### Notebook'u kendin çalıştırmak istersen

Sadece kendi fotoğrafınla denemek için veri setine gerek yok, [aşağıdaki](#kendi-fotoğrafınla-dene-kiyafet_denepy) `kiyafet_dene.py` yeterli. Eğitimi baştan tekrarlamak istersen:

1. [DeepFashion-MultiModal](https://github.com/yumingj/DeepFashion-MultiModal) sayfasındaki **lisans sözleşmesini oku ve kabul et** (yalnızca ticari olmayan araştırma).
2. Aynı sayfadaki resmi linklerden **`labels.zip`** ve **`segm.zip`** dosyalarını indir, kendi Drive'ında `MyDrive/WashMom_Vision/data_raw/` klasörüne yükle.
3. **`images.zip`** (6,8 GB) gerekmez: notebook'un 2. bölümü bu dosyayı resmi linkten `gdown` ile kendisi indirir (dosya yoksa).
4. Notebook'u Google Colab'da (GPU açık) aç ve **Run All** yap. `processed/`, `models/` ve `outputs/` klasörleri otomatik oluşturulur; modeller sıfırdan eğitilir.

## Notebook bölümleri

| Bölüm | İçerik |
|---|---|
| 1-5 | Kurulum, veri erişimi, etiket dosyası, sınıf dağılımı, parsing maskeleri |
| 6-10 | Görüntü + maske, garment crop, etiketlerin elle kontrolü, maske ↔ etiket uyumu, crop kuralı |
| 11-13 | Tüm crop'lar + manifest, EDA grafikleri, alan filtresi, nihai sınıflar, ürün bazlı split ve leakage kontrolü |
| 14 | Custom CNN (baseline): veri hattı, augmentation, class weight, eğitim |
| 15 | EfficientNetV2B0: feature extraction, fine-tuning, yumuşatılmış class weight, karar ayarı (τ) |
| 16 | Test seti: model karşılaştırması, confusion matrix, sınıf bazında F1 |
| 17 | Hata analizi: shortcut kontrolü, confidence eşiği, hata galerisi |
| 18-19 | OpenCV renk analizi (HSV), kural motoru, uçtan uca demo |
| 20 | Sonuç ve proje özeti |
| 21 | Kendi fotoğrafınla dene: GrabCut ile kıyafet maskesi → aynı crop/JPEG adımları → kumaş + renk + yıkama grubu |

## Kendi fotoğrafınla dene (`kiyafet_dene.py`)

Eğitilmiş model repoda hazır (`models/effnet_sqrt_finetuned.keras`, 36,6 MB), GPU gerekmez.

1. Gerekli kütüphaneleri kur (bir kere):
   ```
   pip install tensorflow opencv-python matplotlib numpy
   ```
2. Kıyafet fotoğrafını `my_photos/` klasörüne koy (.jpg / .png).
3. Repo klasöründe terminal aç ve çalıştır:
   ```
   py kiyafet_dene.py my_photos/kiyafet1.jpg
   ```
   (macOS / Linux: `python3 kiyafet_dene.py my_photos/kiyafet1.jpg`)

~20 saniye sonra terminale sonuç yazılır ve fotoğraf / GrabCut maskesi / modele giden crop'u gösteren bir pencere açılır:

```
En olası kumaşlar: cotton 0.98, chiffon 0.02, denim 0.00
{
 "fabric": "cotton",
 "fabric_confidence": 0.98,
 "color_group": "light",
 "washing_group": "WHITE_LIGHT_NORMAL",
 "needs_label_check": false
}
Beyaz ve açık renkli ürünlerle birlikte, normal programda yıkanabilir.
```

Repodaki örnekler (ev ortamında, yerde çekilmiş kendi fotoğraflarım):

| Fotoğraf | Kıyafet | Tahmin | Yıkama grubu |
|---|---|---|---|
| `kiyafet1.jpg` | beyaz tişört | cotton 0,98 | WHITE_LIGHT_NORMAL |
| `kiyafet2.jpg` | kot pantolon | denim 0,99 | DARK_HEAVY |
| `kiyafet3.jpg` | örgü kazak | knitted 0,85 | DELICATE |

Not: Model tek parça kıyafet crop'larıyla eğitildi. En iyi sonuç için fotoğrafta **tek bir kıyafet**, sade bir zeminde ve karenin çoğunu kaplayacak şekilde olmalı. Üst ve alt birlikte görünen kombin fotoğraflarında GrabCut ikisini tek ön plan olarak ayırır ve tahmin bozulabilir.

> **Sorumluluk reddi:** Bu bir öğrenme projesidir; yıkama önerileri yalnızca tahmindir ve model hata yapabilir (test accuracy ≈ %83). Kıyafetinizi yıkamadan önce her zaman **bakım etiketine** uyun. Yazılım "olduğu gibi" sunulur; olası zararlardan sorumluluk kabul edilmez.

## Dataset

DeepFashion-MultiModal — https://github.com/yumingj/DeepFashion-MultiModal
Atıf: Jiang et al., *Text2Human: Text-Driven Controllable Human Image Generation*, SIGGRAPH 2022.

Veri seti yalnızca ticari olmayan araştırma amaçlı kullanılabilir ve lisansı gereği bu repoda **yer almaz** (görüntüler, maskeler, etiketler). Notebook çıktılarındaki veri seti görselleri kaldırıldı, dosya adları `<veri-seti-dosyası>` ile maskelendi. Notebook'u çalıştırmak için veri setini yukarıdaki resmi sayfadan, lisans anlaşmasını kabul ederek indirmek gerekir.

Bu veri setini kullanırsanız lütfen orijinal çalışmaya atıf yapın:

```bibtex
@article{jiang2022text2human,
  title={Text2Human: Text-Driven Controllable Human Image Generation},
  author={Jiang, Yuming and Yang, Shuai and Qiu, Haonan and Wu, Wayne and Loy, Chen Change and Liu, Ziwei},
  journal={ACM Transactions on Graphics (TOG)},
  volume={41},
  number={4},
  articleno={162},
  pages={1--11},
  year={2022},
  publisher={ACM New York, NY, USA},
  doi={10.1145/3528223.3530104}
}
```

## Lisans

- **Bu repodaki kod, notebook, eğitilmiş model ve `my_photos/` fotoğrafları:** [CC BY-NC 4.0](LICENSE). Kaynak göstererek ticari olmayan amaçlarla kullanabilir, değiştirebilir ve paylaşabilirsiniz. **Ticari kullanım yasaktır.**
- **Model (`models/effnet_sqrt_finetuned.keras`):** DeepFashion-MultiModal'dan türetilmiştir. Veri setinin sözleşmesi gereği türetilmiş veriler de ticari amaçla kullanılamaz.
- **Veri seti:** Bu repoya ve bu lisansa dahil değildir. Görüntülerin ve etiketlerin hakları sahiplerine aittir. Bu proje veri setinin yazarlarıyla bağlantılı değildir ve onlar tarafından onaylanmamıştır.
- **EfficientNetV2B0 ImageNet ağırlıkları:** Keras / TensorFlow (Apache 2.0).

Telif satırı ve veri seti notunun tam hâli [NOTICE](NOTICE) dosyasındadır.

Bir hak sahibi olarak bu repodaki herhangi bir içeriğe itirazınız varsa lütfen bir GitHub issue açın; içerik kaldırılacaktır.
