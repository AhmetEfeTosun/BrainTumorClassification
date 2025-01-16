# Brain Tumor Classifaction

Bu proje, Yazılım Geliştirme Laboratuvarı-I dersi kapsamında beyin tümörü tiplerinin bilgisayarlı tomografi (BT) görüntüleri ile sınıflandırılması için kapsamlı bir veri seti oluşturmayı ve farklı derin öğrenme modelleriyle karşılaştırmalı analiz yapmayı amaçlamaktadır.

Projenin ilk aşamasında, menenjiyom, hipofiz tümörü, glioma ve tümörsüz beyin sınıflarını içeren bir veri seti oluşturulmuştur. Veriler, MedPix, OpenI gibi Ulusal Sağlık Enstitüleri (NIH) ve Ulusal Tıp Kütüphanesi (NLM) tarafından geliştirilen tıbbi görüntüleme sitelerinden ve Google Görseller’den elde edilmiştir. Projenin bu kısmına ilişkin kodlara ait bağlantı ekler kısmında verilecektir.

Projenin ikinci aşamasında, Vision Transformer (ViT), ResNet, EfficientNet, DeiT ve Swin Transformer modelleri kullanılarak veri seti üzerinde sınıflandırma çalışmaları gerçekleştirilmiş ve modellerin performansları karşılaştırılmıştır. Bu aşamada, farklı derin öğrenme yaklaşımlarının beyin tümörü sınıflandırmasındaki etkinliği analiz edilmiştir.

Bu kod deposu projenin ikinci kısmına ait kodları barındırmaktadır.

## Gereksinimler

### Python Kurulumu

Bu proje, Python 3.10.9 sürümü ile geliştirilmiştir. Eğer sisteminizde Python yüklü değilse, aşağıdaki bağlantıyı kullanarak en son sürümü indirebilirsiniz:

🔗 Python İndirme Sayfası: https://www.python.org/downloads/

```bash
python --version
```

>Bu adım, Python'un doğru şekilde yüklendiğini kontrol etmek için önemlidir.

### Kütüphaneler

Bu proje için aşağıdaki Python kütüphaneleri gereklidir:

- transformers
- torch
- scikit-learn
- numpy
- matplotlib
- Pillow
- tqdm
- json
- os
- time

```bash
    pip install transformers torch scikit-learn numpy matplotlib Pillow tqdm
```
> **Not:** Gerekli tüm kütüphaneleri yüklemek için yukarıdaki kodu terminalde çalıştırabilirsiniz.


### Projeyi Klonlayın

Öncelikle projeyi kendi bilgisayarınıza klonlayın. Git yüklü değilse, [Git'i buradan](https://git-scm.com/) indirip kurabilirsiniz.
```bash
https://github.com/AhmetEfeTosun/BrainTumorClassification.git
cd BrainTumorClassification
```
>Bu adım, proje dosyalarını yerel makinenize indirir ve proje dizinine geçiş yapar.


------------

>**Not:** Projenin Colab linki ekler kısmında belirtilmiştir.

## Projeyi Çalıştırma

Yukarıdaki kurulumu başarıyla tamamladıysanız, projedeki kodları çalıştırabilirsiniz.

### Kodları Çalıştırabileceğiniz Ortamlar (IDEs)
Kodları aşağıdaki geliştirme ortamlarından (IDEs) birinde açarak çalıştırabilirsiniz:

PyCharm
VS Code
Jupyter Notebook
Google Colab
Bu ortamlardan birinde Vit.py, Resnet.py veya diğer model dosyalarını açarak "Run" (Çalıştır) butonuna basarak veya terminal üzerinden çalıştırabilirsiniz.

### Terminal Üzerinden Çalıştırma
Herhangi bir IDE kullanmadan, doğrudan terminal veya komut istemcisinde (CMD, PowerShell) aşağıdaki komutlarla kodları çalıştırabilirsiniz:

```bash
python Vit.py
```
veya
```bash
python Resnet.py
```

Benzer şekilde, diğer model dosyalarını da aynı şekilde çalıştırabilirsiniz. Modelin eğitimi tamamlandığında, sınıflandırma sonuçları ve performans metrikleri ekrana yazdırılacak ve ilgili dizine kaydedilecektir.

### Proje Dosyalarına İlişkin Açıklama

------------

- **ModelAdı.py (Vit.py, Resnet.py, DeiT.py, EfficientNet.py, Swin.py)**

Modelin eğitilmesi, validasyonu, test edilmesi ve en sonunda grafikler ile JSON dosyası olarak raporlanarak kaydedilmesini sağlar.

- **input-ModelAdı.py (input-vit.py, input-deit.py, input-swin.py)**

İlgili modelin yüklenerek belirtilen resmin ne olduğuna ilişkin model tahminini konsola basar.

# Sonuç
Swin Transformer, üstün doğruluk ve sınıf dengesi ile öne çıkmıştır. ViT ve DeiT modelleri, hız ve doğruluk arasında denge sağlayarak alternatifler arasında dikkat çekmiştir. EfficientNet, hız odaklı projeler için önerilirken, ResNet modelinin performansı iyileştirilmelidir. Modellerin seçiminde proje ihtiyaçlarına göre doğruluk ve hız arasında bir denge gözetilmelidir.
Elde Edilen sonuçların karşılaştırmasını daha detaylı olarak rapor dosyamızda bulabilirsiniz.

## ViT
    1. Doğruluk: %89.19
    2. F1 Skoru: %89.22
    3. AUC: 0.984
![vit_training_validation_plots](https://github.com/user-attachments/assets/61e370ca-a415-4854-bf67-b59347ea6b20)
![vit_roc_curve](https://github.com/user-attachments/assets/0c85ace5-c225-488e-972c-1d1f18c22134)
![vit_normalized_confusion_matrix](https://github.com/user-attachments/assets/75f11b00-4361-4478-92c9-cc150b502b6d)


## Swin
    1. Doğruluk: %90.29
    2. F1 Skoru: %90.24
    3. AUC: 0.986
![WhatsApp Image 2024-12-29 at 01 28 02](https://github.com/user-attachments/assets/865d1d96-8f1a-4e38-90cd-5cc62bd759f1)
![WhatsApp Image 2024-12-29 at 01 28 02 (2)](https://github.com/user-attachments/assets/53de35cb-a306-4a22-aa8d-3aaa00d7ec59)
![WhatsApp Image 2024-12-29 at 01 28 02 (1)](https://github.com/user-attachments/assets/d4fdc186-a8fe-41d1-a245-9ff2a9326654)

## ResNet
    1. Doğruluk: %79.05
    2. F1 Skoru: %78.77
    3. AUC: 0.948
![resnet_training_validation_plots](https://github.com/user-attachments/assets/a5d7d170-8296-41d4-b287-ede6c66fdda3)
![resnet_roc_curve](https://github.com/user-attachments/assets/4a0ddd2c-a6c1-4da1-b36e-26edd125728e)
![resnet_normalized_confusion_matrix](https://github.com/user-attachments/assets/8fc1e683-1608-4f73-ac79-447769edc724)

## EfficientNet
    1. Doğruluk: %87.13
    2. F1 Skoru: %87.13
    3. AUC: 0.979
![efficientnet_training_validation_plots](https://github.com/user-attachments/assets/ac40219a-2c13-4dd9-8b67-023b8fde5da5)
![efficientnet_roc_curve](https://github.com/user-attachments/assets/61dac522-9a4a-41fc-a448-a0445a4bb811)
![efficientnet_normalized_confusion_matrix](https://github.com/user-attachments/assets/815d66a3-cbbd-41a0-bd0a-eb0170a5bf69)

## Deit
    1. Doğruluk: %88.10
    2. F1 Skoru: %88.10
    3. AUC: 0.978
![deit_training_validation_plots](https://github.com/user-attachments/assets/877956ee-42df-4307-a790-1205b41528e4)
![deit_roc_curve](https://github.com/user-attachments/assets/1662419b-a23d-46e3-a001-da7164929ef4)
![deit_normalized_confusion_matrix](https://github.com/user-attachments/assets/1266095f-ec81-4a2b-8ae0-a50062fdb8f9)

> **Not:** Elde ettiğimiz bir çok sonuç burada belirtilmemiştir. Sonuçlara daha detaylı ulaşmak isterseniz raporumuza göz gezdirebilir, Drive bağlantısındaki dosyaları inceleyebilirsiniz.

# EKLER

- Proje Raporu: [Rapor](https://github.com/user-attachments/files/18442124/51_Proje_Raporu.1.2.docx)
- Drive: [Veri Seti](https://drive.google.com/file/d/1Vgpcm6XK3A6hF5fRSx5u_N7SBrsL7OjI/view?usp=sharing)
- Drive: [Elde Edilen Sonuçlar](https://drive.google.com/drive/folders/11QflzbMz7EVOPPI3MI9VDewDsp2JJKgv?usp=sharing)
- Colab: [Colab](https://colab.research.google.com/drive/1Stoaqq2VCA4-twpTYqz4BhbPHRSU2LmK?usp=sharing)

# Hazırlayanlar:
- Ahmet Efe Tosun: [GitHub](https://github.com/AhmetEfeTosun)
- Umut Gülfidan: [GitHub](https://github.com/umutgulfidan)
