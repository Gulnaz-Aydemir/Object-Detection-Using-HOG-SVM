import matplotlib
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
import cv2
from skimage.feature import hog
from skimage import exposure
import os

# DOSYA YOLLARI
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POS_DIR = os.path.join(BASE_DIR, "data", "training_set", "pos")
RESULT_DIR = os.path.join(BASE_DIR, "data", "results")

# Sonuç klasörü yoksa oluştur demek.
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

def visualize_hog():
    # 1. Örnek bir resim bul (pos klasöründen ilk resmi aldım.)
    if not os.path.exists(POS_DIR):
        print("Hata: Eğitim verisi (pos klasörü) bulunamadı.")
        return

    images = [f for f in os.listdir(POS_DIR) if f.endswith(".jpg") or f.endswith(".png")]
    if not images:
        print("Hata: pos klasöründe resim yok.")
        return
    
    # İlk resmi seç
    img_name = images[0]
    img_path = os.path.join(POS_DIR, img_name)
    print(f"İşlenen resim: {img_name}")

    # Resmi Oku
    img = cv2.imread(img_path)
    if img is None:
        print("Resim okunamadı!")
        return
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 64x64 Boyutuna getir
    resized_img = cv2.resize(gray, (64, 64))

    # 2. HOG Özelliklerini Çıkar ve Görselleştir
    fd, hog_image = hog(resized_img, 
                        orientations=9, 
                        pixels_per_cell=(8, 8), 
                        cells_per_block=(2, 2), 
                        visualize=True, 
                        block_norm='L2-Hys')

    # Görseli daha net yapmak için kontrast ayarı için:
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    # 3. Yan Yana Çizdir ve Kaydet demek.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(resized_img, cmap=plt.cm.gray)
    ax1.set_title('Orijinal Resim')

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('HOG (Gradyanlar)')

    # Kaydetmsk için:
    save_path = os.path.join(RESULT_DIR, "hog_visualization.png")
    plt.savefig(save_path)
    print(f"BAŞARILI! HOG görseli şuraya kaydedildi: {save_path}")
    plt.close()

if __name__ == "__main__":
    visualize_hog()