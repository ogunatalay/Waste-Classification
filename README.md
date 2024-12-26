# Waste-Classification
Materyal Tahmini Projesi: Genel Bakış
🎯 Projenin Amacı:
Bu proje, makine öğrenmesi ve derin öğrenme teknikleri kullanarak atık materyallerin (çöp) türlerini görüntü verileri üzerinden sınıflandırmayı amaçlamaktadır. Özellikle geri dönüşüm ve çevre koruma projelerinde kullanılmak üzere geliştirilen bu sistem, bir atık materyalin plastik, cam, kağıt, karton, metal veya genel çöp olup olmadığını tespit eder.

📁 1. Veri Seti:
Veri Seti Yapısı:
Train: Modelin eğitilmesi için kullanılır.
Test: Modelin performansının ölçülmesi için kullanılır.
Kategoriler:
Cardboard (Karton)
Glass (Cam)
Metal
Paper (Kağıt)
Plastic (Plastik)
Trash (Çöp)
Her kategori, ilgili materyale ait görüntüleri içermektedir.

🧠 2. Kullanılan Modeller:
1. CNN (Convolutional Neural Network):
Amaç: Görüntülerin doğrudan özelliklerini çıkararak sınıflandırma yapmak.
Katmanlar:
Conv2D ve MaxPooling2D: Görüntüdeki özellikleri algılamak.
Flatten ve Dense: Özellikleri sınıflandırmak.
Dropout: Aşırı öğrenmeyi (overfitting) önlemek.
2. MobileNet (Transfer Learning):
Amaç: Önceden eğitilmiş bir modelin bilgi birikimini kullanarak daha hızlı ve doğru tahmin yapmak.
Yöntem: MobileNet'in alt katmanları donduruldu, sadece yeni eklenen katmanlar eğitildi.
✅ Avantaj: İki modelin bir arada kullanılması esneklik ve performans açısından etkili bir yöntem oluşturdu.

📈 3. Model Performansı:
Genel Doğruluk: %78
Eğitim ve Doğrulama Doğruluğu:
Eğitim doğruluğu %40'tan %76'ya yükseldi.
Doğrulama doğruluğu %65'ten %78'e çıktı.
Eğitim ve Doğrulama Kaybı:
Eğitim kaybı: 1.999 → 0.698
Doğrulama kaybı: 0.967 → 0.695
✅ Sonuç: Model, eğitim ve doğrulama veri setlerinde istikrarlı bir öğrenme gerçekleştirdi ve aşırı öğrenme (overfitting) göstermedi.

🔍 4. Modelin Tahminleri:
Model, aşağıdaki materyalleri başarılı bir şekilde tespit etti:

Kağıt
Plastik
Metal
Karton
Çöp
Cam
🌍 5. Uygulama Alanları:
Akıllı geri dönüşüm sistemleri
Atık yönetimi ve çevre koruma projeleri
Endüstriyel atık ayrıştırma tesisleri
📝 6. Sonuç:
Bu proje, görüntü tabanlı atık materyal sınıflandırma için etkili bir çözüm sunmaktadır. CNN ve MobileNet modellerinin kombinasyonu, hem eğitim sürecini hızlandırmış hem de model doğruluğunu artırmıştır.

Eğer daha fazla ayrıntı istersen veya teknik detayları incelemek istersen, sorabilirsin!
