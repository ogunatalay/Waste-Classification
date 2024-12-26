📊 Atık Materyal Sınıflandırma ve Karbon Ayak İzi Tahmini Projesi

🎯 Projenin Amacı
Bu projede, atık materyallerin görüntü verileri kullanılarak sınıflandırılması ve malzemelerin çevresel etkilerinin (karbon ayak izi) analiz edilmesi amaçlanmıştır. Proje, makine öğrenmesi (Random Forest, Decision Tree, SVM) ve derin öğrenme (MobileNet ve özel CNN katmanları) teknikleri kullanılarak gerçekleştirilmiştir.

📁 Veri Seti Yapısı ve Hazırlığı
Veri Kaynağı: Google Drive üzerinden erişilen split-garbage-dataset.
Klasör Yapısı:
Train: Modelin öğrenmesi için kullanılan veri seti.
Test: Modelin performansını değerlendirmek için kullanılan veri seti.
Kategoriler:
Plastik
Metal
Kağıt
Karton
Cam
Çöp

Görüntü İşleme:
Görseller yeniden boyutlandırıldı (224x224) ve normalize edildi.
Veri artırma teknikleri uygulandı (döndürme, kaydırma, yakınlaştırma vb.).


🧠 Derin Öğrenme Modelleri
a. MobileNet (Transfer Learning)
Temel Model: MobileNet, önceden eğitilmiş bir ağdır ve imagenet ağırlıkları kullanıldı.
Yapı:
Alt katmanlar donduruldu ve yeniden eğitilmedi.
Üzerine özel CNN katmanları eklendi.
Çıktı katmanı, sınıf sayısına göre (softmax) oluşturuldu.

Eğitim: Model, eğitim veri seti ile eğitildi ve doğrulama veri setiyle değerlendirildi.

Performans Metriği:
Eğitim doğruluğu ve kaybı gözlemlendi.
Test doğruluğu: %78


🤖 Makine Öğrenmesi Modelleri
MobileNet özellikleri çıkarılarak aşağıdaki klasik makine öğrenmesi modelleriyle sınıflandırma yapıldı:

a. Random Forest (Rastgele Ormanlar)
Amaç: Görüntü özelliklerinden sınıflandırma yapmak.
Performans Metriği: Accuracy, Precision, Recall, F1 Score hesaplandı.
b. Decision Tree (Karar Ağaçları)
Amaç: Görüntü özelliklerini ağaç yapısı üzerinden analiz ederek sınıflandırma yapmak.
Performans Metriği: Accuracy, Precision, Recall, F1 Score hesaplandı.
c. SVM (Destek Vektör Makineleri)
Amaç: Özellik uzayında en iyi sınıflandırma sınırını belirlemek.
Performans Metriği: Accuracy, Precision, Recall, F1 Score hesaplandı.
Sonuç: Makine öğrenmesi modelleri üzerinde değerlendirme yapılarak en iyi performans gösteren model belirlendi.

Görselleştirmeler
a. Eğitim ve Test Doğruluğu/Kaybı Grafikleri
Eğitim sürecinde doğruluk ve kayıp metriklerinin değişimi grafiklerle analiz edildi.
b. Karbon Ayak İzi Analizi
Materyallerin karbon ayak izi katkıları görselleştirildi.
Plastik en yüksek karbon ayak izine, çöp ise en düşük karbon ayak izine sahip olarak belirlendi.
c. Scatter Plot ile Malzeme Türleri ve Karbon Ayak İzi Katkısı
Malzeme türleri ve karbon ayak izi katkıları arasında bir scatter plot oluşturuldu.

📸 Görsel Tahmin Örneği
Kullanıcı tarafından sağlanan görüntüler modele verilerek tahmin yapıldı.
Örneğin bir plastik atığın doğru sınıflandırıldığı gözlemlendi.

🌍 Uygulama Alanları
Akıllı Atık Yönetim Sistemleri: Otomatik atık sınıflandırma cihazları.
Çevre Koruma Projeleri: Geri dönüşüm süreçlerinin iyileştirilmesi.
Endüstriyel Kullanım: Atık yönetim tesislerinde hızlı ayrıştırma.


📝 Sonuç
MobileNet tabanlı transfer öğrenme modeli ve makine öğrenmesi algoritmaları başarılı bir şekilde kullanıldı.
Random Forest modeli, en yüksek başarı oranını gösterdi.
Proje, atık yönetimi ve geri dönüşüm sistemlerinde pratik ve etkili bir çözüm sunmaktadır.
