import cv2
import joblib
import os
from skimage.feature import hog

# --- AYARLAR ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "trained_classifier.pkl")
TEST_IMG_DIR = os.path.join(BASE_DIR, "data", "test_images")
RESULT_DIR = os.path.join(BASE_DIR, "data", "results")

# HOG parametreleri (aynısı gibi)
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
WINDOW_SIZE = (64, 64)
STEP_SIZE = 10 # Adım aralığı

if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

def sliding_window(image, step_size, window_size):
    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

def detect_objects():
    # 1. Modeli Yüklemek için:
    if not os.path.exists(MODEL_PATH):
        print("Hata: Model dosyası yok.")
        return
    
    print("Model yükleniyor...")
    model = joblib.load(MODEL_PATH)
    
    # 2. Resimleri Bul
    test_images = [f for f in os.listdir(TEST_IMG_DIR) if f.endswith(('.jpg', '.png'))]
    
    if not test_images:
        print("Klasörde test edilecek resim yok.")
        return

    # 3. Tespit Başlayacak
    for img_name in test_images:
        img_path = os.path.join(TEST_IMG_DIR, img_name)
        print(f"\nİşleniyor: {img_name}")
        
        original_img = cv2.imread(img_path)
        if original_img is None: continue
        
        # Büyük resimleri küçült (Hız için)
        if original_img.shape[1] > 800:
            scale = 0.5
            original_img = cv2.resize(original_img, None, fx=scale, fy=scale)

        gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        detections = [] 

        print("Taranıyor...")
        for (x, y, window) in sliding_window(gray, STEP_SIZE, WINDOW_SIZE):
            if window.shape[0] != WINDOW_SIZE[1] or window.shape[1] != WINDOW_SIZE[0]:
                continue
            
            features = hog(window, orientations=HOG_ORIENTATIONS, 
                           pixels_per_cell=HOG_PIXELS_PER_CELL,
                           cells_per_block=HOG_CELLS_PER_BLOCK, 
                           block_norm='L2-Hys', visualize=False)
            
            prediction = model.predict(features.reshape(1, -1))
            
            if prediction == 1:
                score = model.decision_function(features.reshape(1, -1))[0]
                if score > 0.8: # Eminse ekle
                    detections.append((x, y))

        # Çizmek için:
        for (x, y) in detections:
            cv2.rectangle(original_img, (x, y), (x + WINDOW_SIZE[0], y + WINDOW_SIZE[1]), (0, 255, 0), 2)
            
        save_path = os.path.join(RESULT_DIR, f"result_{img_name}")
        cv2.imwrite(save_path, original_img)
        print(f"Sonuç kaydedildi: {save_path}")

if __name__ == "__main__":
    detect_objects()