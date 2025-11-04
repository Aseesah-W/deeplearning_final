**หัวข้อโครงงาน**

Deep Learning-Based Nail Disease Detection using CNN with Spatial Attention

**ทำไมหัวข้อนี้น่าสนใจ และเหตุผลในการเลือก**

สุขภาพของเล็บสามารถบ่งบอกถึงภาวะสุขภาพภายในร่างกาย เช่น โรคโลหิตจาง การติดเชื้อรา หรือโรคหัวใจ การวิเคราะห์ภาพเล็บด้วยตาเปล่ามักขึ้นอยู่กับประสบการณ์ของแพทย์ผิวหนัง ซึ่งอาจมีความคลาดเคลื่อน การใช้ Deep Learning โดยเฉพาะ Convolutional Neural Network (CNN) จึงเป็นแนวทางที่ช่วยให้อุปกรณ์สามารถ วิเคราะห์โรคจากภาพเล็บได้อัตโนมัติและมีความแม่นยำสูง

หัวข้อนี้น่าสนใจเพราะเป็นการประยุกต์เทคโนโลยี AI ด้านคอมพิวเตอร์วิทัศน์ (Computer Vision) เพื่อช่วยในงานด้านสุขภาพเบื้องต้น ซึ่งอาจพัฒนาเป็นระบบตรวจสุขภาพด้วยกล้องสมาร์ตโฟนในอนาคตได้

**ทำไมต้องใช้ Deep Learning และเปรียบเทียบกับวิธีอื่น**

| **วิธีการ** | **ลักษณะการทำงาน** | **ข้อดี** | **ข้อด้อย** |
| --- | --- | --- | --- |
| Traditional Image Processing | ใช้เทคนิค threshold, edge detection, color segmentation | เข้าใจง่าย, ใช้คำนวณน้อย | ไม่ทนต่อแสง สี หรือรูปแบบที่หลากหลายของเล็บ |
| Classical Machine Learning | ใช้ feature เช่น texture, color แล้วใช้ SVM หรือ Random Forest | ทำงานได้ดีกับข้อมูลที่ชัดเจน | ต้องออกแบบ feature ด้วยมือ (manual feature extraction) |
| Deep Learning (CNN) | ให้โมเดลเรียนรู้ feature เองจากภาพดิบ | มีความแม่นยำสูง, ทำงานอัตโนมัติ, ขยายไปใช้กับภาพจริงได้ดี | ต้องใช้ dataset มากและทรัพยากรสูง |

ดังนั้นการใช้ Deep Learning (CNN) จึงเหมาะสม เพราะสามารถเรียนรู้คุณลักษณะของเล็บโดยไม่ต้องออกแบบ feature เอง และยังสามารถเพิ่มโมดูล Attention เพื่อเน้นบริเวณสำคัญของภาพได้

**สถาปัตยกรรมของโมเดล (Model Architecture)**

โครงสร้างที่ใช้คือ ResNet-18 ซึ่งเป็น CNN ที่มี Residual Block ช่วยให้ gradient ไหลย้อนกลับได้ดีในชั้นลึก และมีการ "เพิ่มส่วนที่ออกแบบเอง" คือ Spatial Attention Module

โมเดล ResNet18WithAttention เริ่มจาก convolution ขนาด 7×7 (3→64) เพื่อดึงลักษณะขั้นต้นตามด้วย batch normalization และ ReLU แล้วลด spatial dimension ด้วย max-pooling ก่อนจะเข้าสู่สเตจของ residual blocks แบบ ResNet-18 ซึ่งประกอบด้วย 4 กลุ่ม (แต่ละกลุ่มมี 2 BasicBlocks) ที่เพิ่มจำนวนฟีเจอร์แผนที่ (feature maps) ตามลำดับเป็น 64 → 128 → 256 → 512 โดยในช่วงที่เปลี่ยนขนาด spatial หรือจำนวน channel จะใช้ downsample (1×1 convolution + BN) ในเส้นทาง skip เพื่อปรับขนาดก่อนการบวกรวมแบบ residual การใช้ residual connection ช่วยให้สัญญาณไหลย้อนกลับได้ดีขึ้นและลดปัญหา vanishing gradient

หลังจากได้ feature map สุดท้ายขนาด 512×7×7 โมเดลจะนำ feature map นั้นไปผ่าน Spatial Attention ที่รวมข้อมูลแบบ channel-wise (max และ average pooling) แล้วประมวลผลด้วย convolution 7×7 ตามด้วย sigmoid เพื่อสร้าง attention mask ขนาด 1×7×7 ซึ่งคูณเข้ากับ feature map ต้นทางอย่างเป็นองค์รวม (channel-wise multiplication) ส่งผลให้โมเดลสามารถเน้นพื้นที่สำคัญทาง spatial ได้ดีขึ้น จากนั้นใช้ adaptive average pooling เพื่อลด spatial dimension เป็น 1×1 ต่อ channel แล้ว flatten เป็นเวกเตอร์ความยาว 512 ส่งต่อเข้า fully-connected layer (512→6) เพื่อทำนายคลาสสุดท้าย

**ภาพรวมสถาปัตยกรรม**

**Activation Function:**

ใช้ ReLU ใน ResNet และ Sigmoid ใน Attention Module เพื่อสร้างค่าความสนใจระหว่าง 0-1

**จำนวนโหนดและการเชื่อมต่อโดยสรุป:**

| **Layer** | **Input Shape** | **Output Shape** | **Activation** |
| --- | --- | --- | --- |
| Conv1 | (3, 224, 224) | (64, 112, 112) | ReLU |
| ResNet Blocks | (64-512, …) | (512, 7, 7) | ReLU |
| Spatial Attention | (512, 7, 7) | (1, 7, 7) | Sigmoid |
| FC Layer | (512,) | (6,) | Softmax |

**อธิบายโค้ด (Code Explanation)**

**Section 1-2:** ติดตั้งไลบรารีและตั้งค่าพารามิเตอร์ เช่น batch size, epochs, image size

\# ===========================

\# 1. Install & Import Packages

\# ===========================

!pip install torch torchvision torchaudio matplotlib seaborn scikit-learn

!pip install kagglehub

import torch

import torch.nn as nn

import torch.optim as optim

from torch.utils.data import DataLoader

from torchvision import datasets, transforms, models

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import confusion_matrix, accuracy_score

import numpy as np

import os

import kagglehub

path = kagglehub.dataset_download("nikhilgurav21/nail-disease-detection-dataset")

print("📁 Dataset downloaded to:", path)

import os

for root, dirs, files in os.walk(path):

&nbsp;   level = root.replace(path, '').count(os.sep)

&nbsp;   indent = ' ' \* 2 \* level

&nbsp;   print(f"{indent}{os.path.basename(root)}/")

&nbsp;   subindent = ' ' \* 2 \* (level + 1)

&nbsp;   for f in files:

&nbsp;       print(f"{subindent}{f}")

\# ===========================

\# 2. Config

\# ===========================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 16

EPOCHS = 10

NUM_CLASSES = 6  # healthy + 5 diseases

IMAGE_SIZE = 224

DATASET_PATH = os.path.join(path, "data")

**Section 3:** เตรียมข้อมูล (Data Augmentation & DataLoader) โดยใช้ torchvision.datasets.ImageFolder และ transforms เพื่อทำ normalization, rotation, color jitter ฯลฯ

\# ===========================

\# 3. Data Augmentation & Loader

\# ===========================

train_transforms = transforms.Compose(\[

&nbsp;   transforms.RandomResizedCrop(IMAGE_SIZE),

&nbsp;   transforms.RandomHorizontalFlip(),

&nbsp;   transforms.RandomRotation(15),

&nbsp;   transforms.ColorJitter(brightness=0.2, contrast=0.2),

&nbsp;   transforms.ToTensor(),

&nbsp;   transforms.Normalize(mean=\[0.485,0.456,0.406\], std=\[0.229,0.224,0.225\])

\])

val_transforms = transforms.Compose(\[

&nbsp;   transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

&nbsp;   transforms.ToTensor(),

&nbsp;   transforms.Normalize(mean=\[0.485,0.456,0.406\], std=\[0.229,0.224,0.225\])

\])

train_dataset = datasets.ImageFolder(os.path.join(DATASET_PATH, "train"), transform=train_transforms)

val_dataset = datasets.ImageFolder(os.path.join(DATASET_PATH, "validation"), transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

class_names = train_dataset.classes

print("Classes:", class_names)

**Section 4:** สร้างคลาส SpatialAttention ซึ่งรวมค่าเฉลี่ยและค่าสูงสุดของแต่ละช่อง จากนั้นใช้ convolution 7×7 เพื่อเน้นบริเวณสำคัญ

\# ===========================

\# 4. Spatial Attention Block

\# ===========================

class SpatialAttention(nn.Module):

&nbsp;   def \__init_\_(self, kernel_size=7):

&nbsp;       super(SpatialAttention, self).\__init_\_()

&nbsp;       self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)

&nbsp;       self.sigmoid = nn.Sigmoid()

&nbsp;   def forward(self, x):

&nbsp;       avg_out = torch.mean(x, dim=1, keepdim=True)

&nbsp;       max_out,_ = torch.max(x, dim=1, keepdim=True)

&nbsp;       x = torch.cat(\[avg_out, max_out\], dim=1)

&nbsp;       x = self.conv(x)

&nbsp;       return self.sigmoid(x)

**Section 5:** สร้างโมเดล ResNet18WithAttention โดยนำ ResNet18 มาตัดส่วน fully connected เดิมออก แล้วเพิ่มโมดูล Attention

\# ===========================

\# 5. ResNet18 + Attention Model

\# ===========================

class ResNet18WithAttention(nn.Module):

&nbsp;   def \__init_\_(self, num_classes=NUM_CLASSES):

&nbsp;       super().\__init_\_()

&nbsp;       base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

&nbsp;       self.features = nn.Sequential(\*list(base_model.children())\[:-2\])  # Remove avgpool & fc

&nbsp;       self.attn = SpatialAttention()

&nbsp;       self.pool = nn.AdaptiveAvgPool2d((1, 1))

&nbsp;       self.fc = nn.Linear(512, num_classes)

&nbsp;   def forward(self, x):

&nbsp;       x = self.features(x)

&nbsp;       attn_map = self.attn(x)

&nbsp;       x = x \* attn_map

&nbsp;       x = self.pool(x)

&nbsp;       x = torch.flatten(x, 1)

&nbsp;       x = self.fc(x)

&nbsp;       return x

model = ResNet18WithAttention().to(DEVICE)

print(model)

**Section 6-7:** สร้าง loss (CrossEntropyLoss) และ optimizer (Adam), จากนั้นเทรนโมเดลพร้อมคำนวณ accuracy

\# ===========================

\# 6. Loss & Optimizer

\# ===========================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=1e-4)

\# ===========================

\# 7. Training Loop

\# ===========================

train_losses, val_losses = \[\], \[\]

train_acc, val_acc = \[\], \[\]

for epoch in range(EPOCHS):

&nbsp;   model.train()

&nbsp;   running_loss, correct, total = 0, 0, 0

&nbsp;   for images, labels in train_loader:

&nbsp;       images, labels = images.to(DEVICE), labels.to(DEVICE)

&nbsp;       optimizer.zero_grad()

&nbsp;       outputs = model(images)

&nbsp;       loss = criterion(outputs, labels)

&nbsp;       loss.backward()

&nbsp;       optimizer.step()

&nbsp;       running_loss += loss.item() \* images.size(0)

&nbsp;       \_, predicted = torch.max(outputs, 1)

&nbsp;       total += labels.size(0)

&nbsp;       correct += (predicted == labels).sum().item()

&nbsp;   train_losses.append(running_loss/total)

&nbsp;   train_acc.append(correct/total)

&nbsp;   # Validation

&nbsp;   model.eval()

&nbsp;   val_loss, correct_val, total_val = 0,0,0

&nbsp;   with torch.no_grad():

&nbsp;       for images, labels in val_loader:

&nbsp;           images, labels = images.to(DEVICE), labels.to(DEVICE)

&nbsp;           outputs = model(images)

&nbsp;           loss = criterion(outputs, labels)

&nbsp;           val_loss += loss.item() \* images.size(0)

&nbsp;           \_, predicted = torch.max(outputs, 1)

&nbsp;           total_val += labels.size(0)

&nbsp;           correct_val += (predicted == labels).sum().item()

&nbsp;   val_losses.append(val_loss/total_val)

&nbsp;   val_acc.append(correct_val/total_val)

&nbsp;   print(f"Epoch \[{epoch+1}/{EPOCHS}\] "

&nbsp;         f"Train Loss: {train_losses\[-1\]:.4f}, Train Acc: {train_acc\[-1\]\*100:.2f}% "

&nbsp;         f"Val Loss: {val_losses\[-1\]:.4f}, Val Acc: {val_acc\[-1\]\*100:.2f}%")

&nbsp;       # 🔹 Show predictions every 2 epochs

&nbsp;   if (epoch+1) % 2 == 0:

&nbsp;       images, labels = next(iter(val_loader))

&nbsp;       images, labels = images.to(DEVICE), labels.to(DEVICE)

&nbsp;       outputs = model(images)

&nbsp;       \_, preds = torch.max(outputs, 1)

&nbsp;       plt.figure(figsize=(12, 6))

&nbsp;       for i in range(6):

&nbsp;           plt.subplot(2, 3, i+1)

&nbsp;           img = images\[i\].cpu().permute(1, 2, 0).numpy()

&nbsp;           img = (img - img.min()) / (img.max() - img.min())

&nbsp;           plt.imshow(img)

&nbsp;           plt.title(f"Pred: {class_names\[preds\[i\]\]}\\nTrue: {class_names\[labels\[i\]\]}")

&nbsp;           plt.axis('off')

&nbsp;       plt.suptitle(f"Validation samples after epoch {epoch+1}")

&nbsp;       plt.show()

**Section 8:** Plot of Training and Validation Loss & Accuracy

\# ===========================

\# 8. Plot Loss & Accuracy

\# ===========================

plt.figure(figsize=(10,4))

\# --- กราฟ Loss ---

plt.subplot(1,2,1)

plt.plot(train_losses, label='Train Loss', marker='o')

plt.plot(val_losses, label='Val Loss', marker='o')

plt.title('Training vs Validation Loss')

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.legend()

\# --- กราฟ Accuracy ---

plt.subplot(1,2,2)

plt.plot(train_acc, label='Train Accuracy', marker='o')

plt.plot(val_acc, label='Val Accuracy', marker='o')

plt.title('Training vs Validation Accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend()

plt.show()

**Section 9:** ประเมินผลด้วย confusion matrix และ visualize attention heatmap เพื่อดูว่าระบบสนใจบริเวณเล็บจริงหรือไม่

\# ===========================

\# 9. Confusion Matrix

\# ===========================

all_preds, all_labels = \[\], \[\]

model.eval()

with torch.no_grad():

&nbsp;   for images, labels in val_loader:

&nbsp;       images = images.to(DEVICE)

&nbsp;       outputs = model(images)

&nbsp;       \_, preds = torch.max(outputs,1)

&nbsp;       all_preds.extend(preds.cpu().numpy())

&nbsp;       all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8,6))

sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")

plt.xlabel("Predicted")

plt.ylabel("True")

plt.title("Confusion Matrix")

plt.show()

from PIL import Image

def visualize_attention(model, image_path, class_names):

&nbsp;   model.eval()

&nbsp;   image = Image.open(image_path).convert("RGB")

&nbsp;   transform = val_transforms

&nbsp;   img_tensor = transform(image).unsqueeze(0).to(DEVICE)

&nbsp;   with torch.no_grad():

&nbsp;       output = model(img_tensor)

&nbsp;       probs = torch.softmax(output, dim=1)

&nbsp;       conf, pred = torch.max(probs, dim=1)

&nbsp;       # attention visualization (ตัวอย่าง)

&nbsp;       features = model.features(img_tensor)

&nbsp;       attn_map = model.attn(features)

&nbsp;       attn_map = attn_map.squeeze().cpu().numpy()

&nbsp;       attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())

&nbsp;   plt.figure(figsize=(10, 5))

&nbsp;   plt.subplot(1, 2, 1)

&nbsp;   plt.imshow(image)

&nbsp;   plt.title(f"Prediction: {class_names\[pred.item()\]} ({conf.item()\*100:.2f}%)")

&nbsp;   plt.subplot(1, 2, 2)

&nbsp;   plt.imshow(image)

&nbsp;   plt.imshow(attn_map, cmap='jet', alpha=0.4)

&nbsp;   plt.title("Attention Heatmap")

&nbsp;   plt.show()

visualize_attention(model, f"{DATASET_PATH}/validation/clubbing/Screen-Shot-2021-10-26-at-12-10-57-PM_png.rf.13d80dc781bd7e9b4d1c2c67ecbacb55.jpg",class_names)

visualize_attention(model, f"{DATASET_PATH}/validation/clubbing/Screen-Shot-2021-10-26-at-12-11-28-PM_png.rf.4f993b88f526fd3ce51dc24eeae5b4d2.jpg",class_names)

visualize_attention(model, f"{DATASET_PATH}/validation/clubbing/Screen-Shot-2021-10-26-at-12-07-03-PM_png.rf.3514795dcb0a1e7bbbd78fd0acc2ac7b.jpg",class_names)

**ลิงค์ที่นำไปสู่โค้ด**

<https://github.com/Aseesah-W/deeplearning_final/blob/main/Deep_Learning_Based_Nail_Health_Analysis.ipynb>

**Dataset และแหล่งที่มา**

- ชื่อ Dataset: Nail Disease Detection Dataset
- แหล่งที่มา: [Kaggle - nikhilgurav21/nail-disease-detection-dataset](https://www.kaggle.com/datasets/nikhilgurav21/nail-disease-detection-dataset)
- ชุดข้อมูลนี้ประกอบด้วยชุดภาพที่ครอบคลุมซึ่งมีจุดมุ่งหมายเพื่อสนับสนุนการพัฒนาโมเดลการเรียนรู้เพื่อตรวจจับโรคเล็บ
- จำนวนคลาส: 6 คลาส ได้แก่ acral_lentiginous_melanoma, healthy_nail, onychogryphosis, blue_finger, clubbing, pitting
- รูปแบบไฟล์: JPEG

**การ Train และ Evaluate โมเดล**

- Hardware: GPU (Colab CUDA)
- Optimizer: Adam (lr=1e-4)
- Loss Function: CrossEntropyLoss
- Batch Size: 16
- Epochs: 10

Epoch \[1/10\] Train Loss: 0.9951, Train Acc: 62.45% Val Loss: 0.3632, Val Acc: 87.91%

Epoch \[2/10\] Train Loss: 0.6242, Train Acc: 77.03% Val Loss: 0.2816, Val Acc: 87.91%

Epoch \[3/10\] Train Loss: 0.5220, Train Acc: 80.80% Val Loss: 0.2957, Val Acc: 90.11%

Epoch \[4/10\] Train Loss: 0.5048, Train Acc: 82.21% Val Loss: 0.2729, Val Acc: 92.31%

Epoch \[5/10\] Train Loss: 0.4578, Train Acc: 83.39% Val Loss: 0.1491, Val Acc: 95.60%

Epoch \[6/10\] Train Loss: 0.3984, Train Acc: 85.63% Val Loss: 0.2788, Val Acc: 85.71%

Epoch \[7/10\] Train Loss: 0.3735, Train Acc: 86.40% Val Loss: 0.3460, Val Acc: 85.71%

Epoch \[8/10\] Train Loss: 0.3787, Train Acc: 86.08% Val Loss: 0.2438, Val Acc: 90.11%

Epoch \[9/10\] Train Loss: 0.3232, Train Acc: 88.33% Val Loss: 0.3461, Val Acc: 89.01%

Epoch \[10/10\] Train Loss: 0.3592, Train Acc: 86.99% Val Loss: 0.3157, Val Acc: 86.81%

Metrics ที่ใช้ประเมิน:

- Accuracy → วัดความถูกต้องโดยรวมของคลาส
- Confusion Matrix → แสดงความสับสนระหว่างแต่ละคลาส
- Visualization → Heatmap ของ Attention Module แสดงว่าระบบเน้นบริเวณเล็บจริง

โมเดล ResNet18 ที่ใช้ Spatial Attention สำหรับการจำแนกโรคเล็บ 6 ประเภท ได้รับการฝึกฝน 10 Epochs โดยมี Validation Accuracy สูงสุดที่ 95.6% ใน Epoch ที่ 5 ก่อนที่จะลดลงและมีความผันผวนไปที่ 86.81% ใน Epoch สุดท้าย

ปัญหาสำคัญที่พบคือ Overfitting: หลังจาก Epoch ที่ 5 เป็นต้นไป โมเดลยังคงเรียนรู้ข้อมูล Training ได้ดี (Training Loss ลดลง, Training Accuracy เพิ่มขึ้น) แต่ประสิทธิภาพบนข้อมูล Validation กลับแย่ลง (Validation Loss เพิ่มขึ้น, Validation Accuracy ลดลงและผันผวน) ซึ่งบ่งชี้ว่าโมเดลจดจำข้อมูล Training มากเกินไปจนไม่สามารถคาดการณ์ข้อมูลใหม่ได้แม่นยำ

จาก Confusion Matrix พบว่าโมเดลทำนายคลาส Healthy_Nail, Acral_Lentiginous_Melanoma, pitting และ blue_finger ได้ดีมาก (มีความแม่นยำสูงถึง 100% ในหลายคลาส) แต่มี ปัญหาในการจำแนกคลาส clubbing และ Onychogryphosis โดยเฉพาะ clubbing ซึ่งมีการทำนายผิดไปหลายคลาสอย่างเห็นได้ชัด

ข้อเสนอแนะหลักคือ: ควรพิจารณาใช้เทคนิค Early Stopping เพื่อหยุดการฝึกฝนเมื่อประสิทธิภาพบนชุด Validation เริ่มลดลง (ประมาณ Epoch ที่ 5) และสำรวจวิธีปรับปรุงประสิทธิภาพสำหรับคลาสที่มีปัญหา เช่น เพิ่มข้อมูลหรือใช้ Regularization

**บทความอ้างอิงและงานที่เกี่ยวข้อง (References)**

ในช่วงหลายปีมานี้ งานวิจัยด้านการวิเคราะห์ภาพเล็บ (nail images) ด้วยเทคนิค deep learning ได้เริ่มมีจำนวนเพิ่มขึ้นอย่างมีนัยสำคัญ เนื่องจากเล็บถือเป็นสัญญาณวินิจฉัยสำคัญของโรคผิวหนัง ภูมิคุ้มกัน และระบบอื่น ๆ ดังนั้น งานวิจัยในสาขานี้สามารถแบ่งออกเป็นกลุ่มหลัก ดังนี้

**1 งานวิจัยเฉพาะด้านเล็บ**

- Autonomous detection of nail disorders using a hybrid capsule CNN (Shandilya et al., 2024) เสนอโมเดล Hybrid Capsule CNN เพื่อจำแนกโรคเล็บ 6 ประเภท ใช้ชุดข้อมูลเดียวกัน (Nail Disease Detection Dataset) และรายงานผล training accuracy ~99.40% และ validation accuracy ~99.25% ซึ่งแสดงถึงศักยภาพของ deep learning ในงานจำแนกโรคเล็บอย่างชัดเจน
- Nail Disease Detection and Classification Using Deep Learning (R. Regin et al., 2022) ใช้ CNN โครงสร้างพื้นฐานและเปรียบเทียบกับวิธี machine learning ดั้งเดิม (SVM, KNN, RF) สำหรับภาพเล็บ โดยเน้นการวิเคราะห์สีและรูปแบบเล็บ
- Assessment of Nail Images for Preliminary disease detection and classification based on CNN: The New Horizon in Disease Detection in Humans (Marulkar & Narain, 2022) ใช้ชุดภาพเล็บจำนวน ~750 ภาพ และโมเดล CNN สำหรับการตรวจโรคเบื้องต้นของเล็บ แสดงให้เห็นถึงการประยุกต์ deep learning เบื้องต้นใน domain เล็บ
- Artificial Intelligence in the Diagnosis of Onychomycosis-Literature Review (บทความทบทวน, 2024) เป็นภาพรวมของ AI/Deep Learning ในงานวินิจฉัยโรคเล็บ เช่น onychomycosis, subungual melanoma โดยระบุว่าโมเดล CNN อย่าง VGG หรือ ResNet ถูกใช้ในหลายงาน และชี้ประเด็นปัญหา เช่น ขาดชุดข้อมูลใหญ่ และขาดการตีความ (explainability)

**2 งานวิจัย Attention Mechanism และโมดูลที่เกี่ยวข้อง**

- CA‑Net: Comprehensive Attention Convolutional Neural Networks for Explainable Medical Image Segmentation (Gu et al., 2020) เสนอโมเดลที่รวมทั้ง Spatial Attention, Channel Attention, และ Scale Attention ในงาน segmentation ทางการแพทย์
- Attention Based Glaucoma Detection: A Large‑scale Database and CNN Model (Li et al., 2019) แสดงการใช้ Spatial Attention module กับภาพ fundus สำหรับตรวจ glaucoma ซึ่งเป็นตัวอย่างดีของการนำ Attention มาใช้ในงานภาพทางการแพทย์

**3 จุดที่ต่อยอด**

จากงานวิจัยดังกล่าว จะเห็นได้ว่างานด้านเล็บส่วนใหญ่ใช้ CNN แบบพื้นฐาน ซึ่งอาจยัง ไม่ได้ใช้โมดูล Attention ทาง spatial โดยเฉพาะ และงานด้าน Attention ในภาพทางการแพทย์แสดงให้เห็นว่า Spatial (และ Channel) Attention ช่วยให้โมเดล "โฟกัส" บริเวณสำคัญของภาพได้ดีขึ้น  
ดังนั้น การใช้ ResNet-18 + Spatial Attention Module ถือว่าเป็นการต่อยอดที่มีความใหม่ โดยเฉพาะใน domain "ภาพเล็บ" ที่ยังมีงานน้อย และโมดูล Attention ที่ใช้ช่วยเน้นบริเวณเล็บ (nail area) อย่างชัดเจน
