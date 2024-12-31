from transformers import  AutoImageProcessor, AutoModelForImageClassification
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_curve, auc , classification_report , roc_auc_score
import torch
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.preprocessing import label_binarize
import time

# Klasör yollarını belirtin
train_dir = 'C:\\Users\\PC\\Desktop\\YazlabDeneme2\\train'
val_dir = 'C:\\Users\\PC\\Desktop\\YazlabDeneme2\\val'
test_dir = 'C:\\Users\\PC\\Desktop\\YazlabDeneme2\\test'

# Sınıf isimlerini algıla
class_names = sorted(os.listdir(train_dir))
num_classes = len(class_names)

# ResNet-50 için işlemci
processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")

# ResNet-50 modeli yükleniyor
model = AutoModelForImageClassification.from_pretrained(
    "microsoft/resnet-50",
    num_labels=num_classes,
    id2label={i: label for i, label in enumerate(class_names)},
    label2id={label: i for i, label in enumerate(class_names)},
    ignore_mismatched_sizes=True
)
# Özel Dataset sınıfı
class ImageDataset(Dataset):
    def __init__(self, root_dir, processor, class_names):
        self.root_dir = root_dir
        self.processor = processor
        self.image_paths = []
        self.labels = []

        for label, class_name in enumerate(class_names):
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.image_paths.append(img_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path)
        inputs = self.processor(images=image, return_tensors="pt", size={"shortest_edge": 224})

        return inputs['pixel_values'][0], label

# Dataset ve DataLoader
train_dataset = ImageDataset(train_dir, processor, class_names)
val_dataset = ImageDataset(val_dir, processor, class_names)
test_dataset = ImageDataset(test_dir, processor, class_names)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Eğitim için optimizer ve loss fonksiyonu
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Cihaz kontrolü
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Gpu: "+str(torch.cuda.is_available()))
model.to(device)

# Eğitim döngüsü
train_accuracies, val_accuracies = [], []
train_losses, val_losses = [], []

# Eğitim süresi başlangıcı
start_train_time = time.time()

for epoch in range(5):
    model.train()
    total_loss, correct_preds, total_preds = 0, 0, 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(pixel_values=images, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=-1)
        correct_preds += (preds == labels).sum().item()
        total_preds += labels.size(0)

    train_accuracies.append(correct_preds / total_preds)
    train_losses.append(total_loss / len(train_loader))

    # Validation
    model.eval()
    val_loss, correct_preds, total_preds = 0, 0, 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            outputs = model(pixel_values=images, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            val_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            correct_preds += (preds == labels).sum().item()
            total_preds += labels.size(0)

    val_accuracies.append(correct_preds / total_preds)
    val_losses.append(val_loss / len(val_loader))

    print(f"Epoch {epoch+1}, Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accuracies[-1]:.4f}")

# Eğitim süresi sonu
end_train_time = time.time()
training_time = end_train_time - start_train_time


# Test verisini değerlendirme
model.eval()
all_preds, all_labels, all_probs = [], [], []
total_test_loss, correct_preds, total_preds = 0, 0, 0

# Çıkarım süresi başlangıcı
start_inference_time = time.time()

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Test Evaluation"):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        outputs = model(pixel_values=images, labels=labels)  # Loss'u hesaplarken labels'i geçiyoruz
        loss = outputs.loss
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)

        total_test_loss += loss.item()
        preds = torch.argmax(probs, dim=-1)
        correct_preds += (preds == labels).sum().item()
        total_preds += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

# Çıkarım süresi sonu
end_inference_time = time.time()
inference_time = end_inference_time - start_inference_time


# Ortalama test loss ve doğruluk hesaplama
test_loss = total_test_loss / len(test_loader)
test_accuracy = correct_preds / total_preds

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Inference Time: {end_inference_time - start_inference_time:.2f} seconds")

# Modeli kaydet
model_save_path = "resnet_model.pth"
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,  # Son eğitim epoch'u
    'train_loss': train_losses,
    'val_loss': val_losses,
    'train_accuracy': train_accuracies,
    'val_accuracy': val_accuracies
}, model_save_path)

print(f"Model başarıyla kaydedildi: {model_save_path}")

#Confusion Matrix oluşturma
cm = confusion_matrix(all_labels, all_preds)

# Hesaplanan değerler
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')
f_score = f1_score(all_labels, all_preds, average='weighted')

# AUC hesaplama
all_labels_binarized = label_binarize(all_labels, classes=list(range(num_classes)))
all_probs_array = np.array(all_probs)
auc_value = roc_auc_score(all_labels_binarized, all_probs_array, average="weighted", multi_class="ovr")

# Sınıf bazlı sensitivite ve spesifisite hesaplama
class_metrics = {}
for i in range(num_classes):
    TP = cm[i, i]  # Doğru pozitifler
    FN = cm[i, :].sum() - TP  # Yanlış negatifler
    FP = cm[:, i].sum() - TP  # Yanlış pozitifler
    TN = cm.sum() - (TP + FN + FP)  # Doğru negatifler
    
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    
    class_metrics[class_names[i]] = {
        "sensitivity": sensitivity,
        "specificity": specificity
    }

# Sınıf bazlı Precision, Recall, F1-Score ve Support
report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)

# Test veri boyutu
test_size = len(test_dataset)
# Ortalama inference time hesaplama (test veri boyutuna bölme)
average_inference_time = inference_time / test_size

# JSON formatında çıktılar
metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f_score": f_score,
    "auc": auc_value,
    "class_metrics": class_metrics,
    "classification_report": report,
    "training_time":training_time,
    "inference_time":inference_time,
    "average_inference_time":average_inference_time
}

# JSON dosyasına yazma
with open('resnet_model_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)


# Eğitim ve test kayıplarını ve doğrulukları çizme
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
plt.plot(range(1, len(val_losses) + 1), val_losses, label="Val Loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label="Train Accuracy")
plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label="Val Accuracy")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Epochs')
plt.legend()

plt.tight_layout()
plt.savefig("resnet_training_validation_plots.png")


# Confusion Matrix'i normalize etme
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Her satırın toplamına böler

print("Normalized Confusion Matrix:")
print(cm_normalized)

# Normalized Confusion Matrix görselleştirmesi
plt.figure()
plt.imshow(cm_normalized, cmap=plt.cm.Blues)
plt.colorbar()
plt.title("Normalized Confusion Matrix")
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.xticks(np.arange(num_classes), class_names, rotation=45)
plt.yticks(np.arange(num_classes), class_names)

# Hücrelerin içerisine sayıları ekleme
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, f'{cm_normalized[i, j]:.2f}', ha='center', va='center')

plt.tight_layout()
plt.savefig("resnet_normalized_confusion_matrix.png")

# ROC eğrisi ve AUC hesaplama
fpr, tpr, _ = roc_curve(all_labels_binarized.ravel(), all_probs_array.ravel())
roc_auc = auc(fpr, tpr)

# ROC Eğrisini çizme
plt.figure(figsize=(10, 8))

for i in range(num_classes):
    # Her sınıf için ROC eğrisini çizme
    fpr, tpr, _ = roc_curve(all_labels_binarized[:, i], np.array(all_probs)[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {class_names[i]} (AUC = {roc_auc:.2f})')

# Yüksek doğru ve düşük yanlış pozitif oranları ile ideal ROC eğrisini ekleme
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Each Class')
plt.legend(loc='lower right')

# Grafik kaydetme
plt.tight_layout()
plt.savefig("resnet_roc_curve.png")

# Gösterme
plt.show()


