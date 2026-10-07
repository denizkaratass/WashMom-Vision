# WashMom Vision - kendi kıyafet fotoğrafınla dene (bilgisayarda, Colab'sız)
# Kullanım:  py kiyafet_dene.py fotograf.jpg
# Gerekli:   models/effnet_sqrt_finetuned.keras  (repoda hazır)
# Notebook'un 21. bölümüyle aynı adımlar: GrabCut maske -> crop -> JPEG -> EfficientNet -> renk -> kural motoru
# Model DeepFashion-MultiModal ile eğitildi: https://github.com/yumingj/DeepFashion-MultiModal (Jiang et al., Text2Human, SIGGRAPH 2022; yalnızca ticari olmayan kullanım)
# Lisans: CC BY-NC 4.0 (LICENSE dosyasına bakın) - ticari kullanım yasaktır.
import sys, os, json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

CLASSES = ["cotton", "denim", "chiffon", "knitted", "leather", "furry"]
TRAIN_COUNTS = [10351, 3402, 1290, 797, 440, 127]                  # train setindeki sınıf sayıları (13. bölüm)
balanced = [sum(TRAIN_COUNTS) / (len(CLASSES) * n) for n in TRAIN_COUNTS]   # compute_class_weight("balanced") formülü
eff_w = {k: float(np.sqrt(v)) for k, v in enumerate(balanced)}    # seçilen modelin sqrt ağırlıkları (15.3)
EFF_TAU, CONF_THRESHOLD = 0.3, 0.55                                 # validation ile seçilen değerler (15.4, 17)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "effnet_sqrt_finetuned.keras")

WASH_TEXT = {
    "DARK_HEAVY": "Koyu renkli ve dayanıklı ürünlerle birlikte yıkanabilir. Hassas ve açık renkli ürünlerle yıkanması önerilmez.",
    "WHITE_LIGHT_NORMAL": "Beyaz ve açık renkli ürünlerle birlikte, normal programda yıkanabilir.",
    "DARK_COLOR_NORMAL": "Koyu ve renkli ürünlerle birlikte, normal programda yıkanabilir. Beyazlarla yıkanması önerilmez.",
    "DELICATE": "Hassas ürün: hassas programda veya elde, benzer hassas ürünlerle yıkanmalı.",
    "LABEL_CHECK": "Özel bakım gerektirebilir: yıkamadan önce bakım etiketini kontrol edin.",
    "USER_CONFIRM": "Model kumaştan emin değil: lütfen kumaş türünü onaylayın.",
}

# --- Notebook'taki fonksiyonların aynısı (10, 15.4, 18, 19, 21. bölümler) ---
def make_crop(img, mask, codes, pad_ratio=0.08, size=224):
    part = np.isin(mask, codes)                        # bu parçaya ait pikseller True
    if part.sum() < 50:                                # parça yok ya da birkaç piksellik gürültü
        return None
    ys, xs = np.where(part)
    y1, y2, x1, x2 = ys.min(), ys.max(), xs.min(), xs.max()          # bounding box
    pad_y, pad_x = int((y2 - y1) * pad_ratio), int((x2 - x1) * pad_ratio)
    y1, y2 = max(0, y1 - pad_y), min(mask.shape[0], y2 + 1 + pad_y)
    x1, x2 = max(0, x1 - pad_x), min(mask.shape[1], x2 + 1 + pad_x)
    clean = img.copy()
    clean[~part] = 255                                 # kıyafet olmayan pikseller beyaz
    crop = clean[y1:y2, x1:x2]
    h, w = crop.shape[:2]
    s = max(h, w)
    square = np.full((s, s, 3), 255, np.uint8)         # beyaz kare tuval
    top, left = (s - h) // 2, (s - w) // 2
    square[top:top + h, left:left + w] = crop
    return cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)

def adjust(probs, weights, tau):
    w = np.array([weights[k] for k in range(len(CLASSES))])
    adj = probs * np.power(w, -tau)                    # ağır sınıfların payını azalt
    return adj / adj.sum(axis=1, keepdims=True)

def color_group(img, mask, codes):
    pixels = img[np.isin(mask, codes)]                 # sadece kıyafet pikselleri (RGB)
    hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    s, v = float(np.median(hsv[:, 1])), float(np.median(hsv[:, 2]))
    if v < 90:
        group = "dark"
    elif s < 35:
        group = "white" if v >= 200 else ("light" if v >= 140 else "dark")
    elif v >= 190 and s < 80:
        group = "light"
    else:
        group = "colored"
    return group, s, v

def washing_group(fabric, color):
    if fabric in ("leather", "furry"):
        return "LABEL_CHECK"
    if fabric in ("knitted", "chiffon"):
        return "DELICATE"
    if color in ("white", "light"):
        return "WHITE_LIGHT_NORMAL"
    return "DARK_HEAVY" if fabric == "denim" else "DARK_COLOR_NORMAL"

def garment_mask(img):
    # GrabCut: kenardan %5 içerideki dikdörtgenin içinde kıyafeti arka plandan ayırır
    h, w = img.shape[:2]
    rect = (int(w * 0.05), int(h * 0.05), int(w * 0.90), int(h * 0.90))
    gc = np.zeros((h, w), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(img, gc, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    return np.isin(gc, [cv2.GC_FGD, cv2.GC_PR_FGD]).astype(np.uint8)   # 1 = kıyafet, 0 = arka plan

def main(path):
    if not os.path.exists(MODEL_PATH):
        sys.exit(f"Model bulunamadı: {MODEL_PATH}\nRepodaki models/effnet_sqrt_finetuned.keras dosyası eksik; repoyu yeniden indir.")
    bgr = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR) if os.path.exists(path) else None  # Türkçe karakterli yol için
    if bgr is None:
        sys.exit(f"Fotoğraf okunamadı, yolu ve formatı (.jpg/.png) kontrol et: {path}")
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    scale = 800 / max(img.shape[:2])                   # GrabCut büyük resimde yavaş
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    m = garment_mask(img)
    crop = make_crop(img, m, [1])
    if crop is None:
        sys.exit("Kıyafet bulunamadı: daha sade bir zeminde, kıyafet ortada olacak şekilde dene.")
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))   # eğitimdeki gibi JPEG
    crop = tf.io.decode_jpeg(buf.tobytes(), channels=3).numpy()

    model = keras.models.load_model(MODEL_PATH)
    probs = adjust(model.predict(crop[None].astype("float32"), verbose=0), eff_w, EFF_TAU)[0]
    fabric, conf = CLASSES[int(probs.argmax())], float(probs.max())
    color, _, _ = color_group(img, m, [1])
    group = washing_group(fabric, color) if conf >= CONF_THRESHOLD else "USER_CONFIRM"
    result = {"fabric": fabric, "fabric_confidence": round(conf, 2), "color_group": color,
              "washing_group": group, "needs_label_check": group in ("LABEL_CHECK", "USER_CONFIRM")}

    top3 = np.argsort(probs)[::-1][:3]
    print("En olası kumaşlar:", ", ".join(f"{CLASSES[k]} {probs[k]:.2f}" for k in top3))
    print(json.dumps(result, ensure_ascii=False, indent=1))
    print(WASH_TEXT[group])

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, im, title in zip(axes, [img, m, crop], ["Fotoğraf", "Maske (GrabCut)", "Modele giden crop"]):
        ax.imshow(im, cmap="gray" if im.ndim == 2 else None)
        ax.set_title(title)
        ax.axis("off")
    plt.suptitle(f"{fabric} ({conf:.2f}) · {color} · {group}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Kullanım: py kiyafet_dene.py kiyafet1.jpg")
    main(sys.argv[1])
