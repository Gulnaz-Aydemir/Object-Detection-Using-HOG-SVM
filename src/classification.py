import os
import cv2
import numpy as np
from skimage.feature import hog
import joblib
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# KLASÖR YOLLARI:
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Proje ana dizini
POS_PATH = os.path.join(BASE_DIR, "data", "training_set", "pos")
NEG_PATH = os.path.join(BASE_DIR, "data", "training_set", "neg")
MODEL_PATH = os.path.join(BASE_DIR, "models", "trained_classifier.pkl")

# HOG AYARLARI
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
IMG_SIZE = (64, 64)

def load_data_and_extract_features():
    data = []
    labels = []
    
    # Klasörlerin varlığını kontrol ettim.
    if not os.path.exists(POS_PATH) or not os.path.exists(NEG_PATH):
        print("HATA: Veri klasörleri bulunamadı! Lütfen önce resimleri indirin.")
        return [], []

    print("Veriler yükleniyor ve özellikler çıkarılıyor...")

    # 1. POS (Araba) Resimlerini İşlemek için:
    print(f" - {POS_PATH} klasörü okunuyor...")
    for filename in os.listdir(POS_PATH):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(POS_PATH, filename)
            img = cv2.imread(img_path)
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, IMG_SIZE)
                features = hog(resized, orientations=HOG_ORIENTATIONS, 
                               pixels_per_cell=HOG_PIXELS_PER_CELL,
                               cells_per_block=HOG_CELLS_PER_BLOCK, 
                               block_norm='L2-Hys', visualize=False)
                data.append(features)
                labels.append(1)

    # 2. NEG (Manzara) Resimlerini İşledim. -> Etiket: 0
    print(f" - {NEG_PATH} klasörü okunuyor...")
    for filename in os.listdir(NEG_PATH):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(NEG_PATH, filename)
            img = cv2.imread(img_path)
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, IMG_SIZE)
                features = hog(resized, orientations=HOG_ORIENTATIONS, 
                               pixels_per_cell=HOG_PIXELS_PER_CELL,
                               cells_per_block=HOG_CELLS_PER_BLOCK, 
                               block_norm='L2-Hys', visualize=False)
                data.append(features)
                labels.append(0)

    return np.array(data), np.array(labels)

def train_model():
    # Verileri hazırlamak için:
    X, y = load_data_and_extract_features()
    
    if len(X) == 0:
        print("Veri bulunamadı, işlem iptal edildi.")
        return

    # Eğitim ve Test olarak ayır (%80 eğitim, %20 test)
    print(f"Toplam {len(X)} veri var. Eğitim ve test olarak ayrılıyor...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # SVM Modelini Eğitmek için:
    print("SVM Modeli eğitiliyor...")
    model = LinearSVC(random_state=42)
    model.fit(X_train, y_train)

    # Test etmek için:
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"\nModel Doğruluğu (Accuracy): %{acc * 100:.2f}")
    print("\nSınıflandırma Raporu:\n", classification_report(y_test, predictions))

    # Modeli Kaydetmek için:
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True) # models klasörü yoksa oluştur
    joblib.dump(model, MODEL_PATH)
    print(f"Model başarıyla kaydedildi: {MODEL_PATH}")

if __name__ == "__main__":
    train_model()