# HOG ile Nesne Tespiti ve Sınıflandırma Projesi

Bu proje, Bilgisayarlı Görü dersinde geliştirilmiştir. Projenin amacı, Histogram of Oriented Gradients (HOG) yöntemini kullanarak görüntülerden öznitelik çıkarımı yapmak ve Destek Vektör Makineleri (SVM) algoritması ile eğitilen bir model sayesinde nesne tespiti (araç ve yaya) gerçekleştirmektir.

Proje; veri setinin hazırlanması, modelin eğitilmesi ve kayan pencere (sliding window) yöntemiyle test edilmesini kapsar.

# Kurulum 
Projenin sorunsuz çalışabilmesi için gerekli Python kütüphanelerinin yüklü olması gerekmektedir.

1. Sanal Ortam (Önerilen): Projeyi çalıştırmadan önce bir sanal ortam (venv) oluşturmanız önerilir.

2. Kütüphanelerin Yüklenmesi: Proje dizininde terminali açarak aşağıdaki komutu çalıştırınız. Bu komut requirements.txt dosyasındaki tüm paketleri otomatik olarak kuracaktır:


pip install -r requirements.txt
Kullanılan Temel Kütüphaneler:

opencv-python: Görüntü işleme işlemleri için.

scikit-image: HOG algoritması için.

scikit-learn: SVM sınıflandırma modeli için.

joblib: Eğitilen modeli kaydetmek ve yüklemek için.

matplotlib: Sonuçları görselleştirmek için.

numpy: Sayısal işlemler için.

# Projenin Çalıştırılması
Proje üç temel problem aşamasından oluşmaktadır. Aşağıdaki adımları sırasıyla uygulayarak projeyi test edebilirsiniz.

1. Adım: HOG Görselleştirme (Problem 1)
HOG algoritmasının bir görüntü üzerinde gradyanları nasıl hesapladığını görmek için bu kodu çalıştırın.

python src/hog_implementation.py
İşlem: data/training_set/pos klasöründen örnek bir araç resmi alır, HOG özniteliklerini hesaplar.

Sonuç: data/results/hog_visualization.png dosyası oluşturulur.

2. Adım: Modelin Eğitilmesi (Problem 3)
Eğitim setindeki (Training Set) verileri kullanarak SVM modelini eğitmek için bu kodu çalıştırın.


python src/classification.py
İşlem: data/training_set klasöründeki tüm resimleri (Pozitif ve Negatif) okur, HOG özelliklerini çıkarır, modeli eğitir ve başarım oranını ekrana yazar.

Sonuç: models/trained_classifier.pkl dosyası oluşturulur. (Hazır model zaten projede mevcuttur, bu adım modeli yeniden eğitir).

3. Adım: Nesne Tespiti Testi (Problem 2)
Eğitilen modeli kullanarak, daha önce görülmemiş test resimlerinde nesne (araba/insan) tespiti yapmak için bu kodu çalıştırın.

Önemli: Test etmek istediğiniz .jpg formatındaki resimleri data/test_images/ klasörüne koyduğunuzdan emin olun.


python src/object_detection.py
İşlem: Kayan pencere (Sliding Window) yöntemi ile resim taranır. Eğitilen model, pencerelerdeki görüntüleri sınıflandırır. Araba veya insan tespit edilen bölgeler yeşil kutu ile çizilir.

Sonuç: İşlenmiş resimler data/results/ klasörüne result_ öneki ile kaydedilir.

Not:
Hazır Model: Proje klasöründe önceden eğitilmiş models/trained_classifier.pkl dosyası mevcuttur. Eğitim adımını (Adım 2) atlayıp doğrudan test adımına (Adım 3) geçebilirsiniz.

Rapor: Projenin teorik detayları, kullanılan yöntemler ve sonuç yorumları report/report.pdf dosyasında sunulmuştur.

Görseller: Raporda kullanılan figürlerin orijinalleri report/figures/ klasöründe yer almaktadır.

Hazırlayan: Gülnaz Aydemir


