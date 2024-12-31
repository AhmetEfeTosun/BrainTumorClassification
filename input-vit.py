from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch
import os

# Test için kullanılan sınıf adlarını içeren listeyi yükleyin
train_dir = 'C:\\Users\\PC\\Desktop\\YazlabDeneme\\train'
class_names = sorted(os.listdir(train_dir))  # Sınıf isimlerini al

# num_classes'ı ve id2label, label2id sözlüklerini oluşturun
num_classes = len(class_names)
id2label = {i: label for i, label in enumerate(class_names)}
label2id = {label: i for i, label in enumerate(class_names)}

# İşlemciyi yükleyin
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

# Modeli yükleyin
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224-in21k',
    num_labels=num_classes,
    id2label=id2label,
    label2id=label2id
)

# Optimizer'ı tanımlayın (örnek olarak Adam optimizer'ı)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Kaydedilen modelin ağırlıklarını yükleyin
checkpoint = torch.load("vit_model.pth")  # Model ağırlıkları
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
train_losses = checkpoint['train_loss']
val_losses = checkpoint['val_loss']
train_accuracies = checkpoint['train_accuracy']
val_accuracies = checkpoint['val_accuracy']

# Modeli değerlendirme moduna alın
model.eval()

# Görseli yükleyin
file_path = "C:\\Users\\PC\\Desktop\\YazlabDeneme\\test\\PituitaryTumor\\resized_adjusted_pituitary1378.jpg"  # Test görseli
image = Image.open(file_path)

# Görseli ön işleyin
inputs = processor(images=image, return_tensors="pt")

# Modeli kullanarak tahmin yap
with torch.no_grad():
    outputs = model(**inputs)

# Çıktıyı işlem
logits = outputs.logits
predicted_class_idx = torch.argmax(logits, dim=-1).item()

# Tahmin edilen sınıf adını al
predicted_class_name = id2label[predicted_class_idx]

print(f"Predicted class name: {predicted_class_name}")
