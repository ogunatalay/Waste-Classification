📊 **Atık Materyal Sınıflandırma ve Karbon Ayak İzi Tahmini Projesi** <br>
**ogun.atalay33@gmail.com** <br>
🎯 **Projenin Amacı** <br>
Bu projede, atık materyallerin görüntü verileri kullanılarak sınıflandırılması ve malzemelerin çevresel etkilerinin (karbon ayak izi) analiz edilmesi amaçlanmıştır. Proje, makine öğrenmesi (Random Forest, Decision Tree, SVM) ve derin öğrenme (MobileNet ve özel CNN katmanları) teknikleri kullanılarak gerçekleştirilmiştir.

📁 **Veri Seti Yapısı ve Hazırlığı** <br>
Veri Kaynağı: Kaggle üzerinden erişilen split-garbage-dataset.
Klasör Yapısı:
I.  Train: Modelin öğrenmesi için kullanılan veri seti.<br>
II. Test: Modelin performansını değerlendirmek için kullanılan veri seti.<br>
Kategoriler:<br>
- Plastik
- Metal
- Kağıt
- Karton
- Cam
- Çöp

Görüntü İşleme:
- Görseller yeniden boyutlandırıldı (224x224) ve normalize edildi.
- Veri artırma teknikleri uygulandı (döndürme, kaydırma, yakınlaştırma vb.).


🧠 Derin Öğrenme Modelleri<br>
a. MobileNet (Transfer Learning) <br>
Temel Model: MobileNet, önceden eğitilmiş bir ağdır ve imagenet ağırlıkları kullanıldı.<br>
Yapı:
- Alt katmanlar donduruldu ve yeniden eğitilmedi.
- Üzerine özel CNN katmanları eklendi.
- Çıktı katmanı, sınıf sayısına göre (softmax) oluşturuldu.

Eğitim: Model, eğitim veri seti ile eğitildi ve doğrulama veri setiyle değerlendirildi.<br>

Performans Metriği:<br>
Eğitim doğruluğu ve kaybı gözlemlendi.<br>
Test doğruluğu: %78<br>


🤖 Makine Öğrenmesi Modelleri<br>
MobileNet özellikleri çıkarılarak aşağıdaki klasik makine öğrenmesi modelleriyle sınıflandırma yapıldı:

a. Random Forest (Rastgele Ormanlar)<br>
Amaç: Görüntü özelliklerinden sınıflandırma yapmak.<br>
Performans Metriği: Accuracy, Precision, Recall, F1 Score hesaplandı.<br>

b. Decision Tree (Karar Ağaçları)<br>
Amaç: Görüntü özelliklerini ağaç yapısı üzerinden analiz ederek sınıflandırma yapmak.<br>
Performans Metriği: Accuracy, Precision, Recall, F1 Score hesaplandı.<br>

c. SVM (Destek Vektör Makineleri)<br>
Amaç: Özellik uzayında en iyi sınıflandırma sınırını belirlemek.<br>
Performans Metriği: Accuracy, Precision, Recall, F1 Score hesaplandı.<br>

Sonuç: Makine öğrenmesi modelleri üzerinde değerlendirme yapılarak en iyi performans gösteren model belirlendi.<br>

Görselleştirmeler<br>
a. Eğitim Doğruluğu/Kaybı Grafikleri<br>
Eğitim sürecinde doğruluk ve kayıp metriklerinin değişimi grafiklerle analiz edildi.<br>
b. Karbon Ayak İzi Analizi<br>
Materyallerin karbon ayak izi katkıları görselleştirildi.<br>
Plastik en yüksek karbon ayak izine, çöp ise en düşük karbon ayak izine sahip olarak belirlendi.<br>
c. Scatter Plot ile Malzeme Türleri ve Karbon Ayak İzi Katkısı<br>
Malzeme türleri ve karbon ayak izi katkıları arasında bir scatter plot oluşturuldu.<br>

📸 Görsel Tahmin Örneği<br>
Kullanıcı tarafından sağlanan görüntüler modele verilerek tahmin yapıldı.<br>
Örneğin bir plastik atığın doğru sınıflandırıldığı gözlemlendi.<br>

🌍 Uygulama Alanları<br>
Akıllı Atık Yönetim Sistemleri: Otomatik atık sınıflandırma cihazları.<br>
Çevre Koruma Projeleri: Geri dönüşüm süreçlerinin iyileştirilmesi.<br>
Endüstriyel Kullanım: Atık yönetim tesislerinde hızlı ayrıştırma.<br>


📝 Sonuç<br>
MobileNet tabanlı transfer öğrenme modeli ve makine öğrenmesi algoritmaları başarılı bir şekilde kullanıldı.<br>
Random Forest modeli, klasik modeller arasında en yüksek başarı oranını gösterdi.<br>
Proje, atık yönetimi ve geri dönüşüm sistemlerinde pratik ve etkili bir çözüm sunmaktadır.<br>
