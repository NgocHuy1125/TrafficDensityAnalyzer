import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from scipy.stats import gaussian_kde
from ultralytics import YOLO
import cv2
import glob
import shutil
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd

# ----- CẤU HÌNH -----
DATA_DIR = './traffic_analysis_output/processed_for_cnn'
NUM_CLASSES = 3  # Low, Medium, High (chỉ dùng 3 class để train)
BATCH_SIZE = 16
IMAGE_SIZE = 224
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
YOLO_MODEL_NAME = 'yolov8n.pt'
VEHICLE_CLASS_IDS = [2, 3, 5, 7]  # COCO: car, motorcycle, bus, truck
YOLO_CONFIDENCE_THRESHOLD = 0.3  # Tăng để cải thiện phát hiện
COUNT_THRESHOLD_LOW_TO_MEDIUM = 5
COUNT_THRESHOLD_MEDIUM_TO_HIGH = 15

# Thiết lập logging
logging.basicConfig(filename='kde_warnings.log', level=logging.WARNING, 
                    format='%(asctime)s - %(message)s')

# ----- CUSTOM DATASET: KDE THỰC TẾ -----
class TrafficDatasetWithKDE(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, yolo_model=None, include_unlabeled=False):
        self.dataset = datasets.ImageFolder(root=root_dir, transform=transform)

        # Chỉ có 3 lớp thực sự train
        self.class_to_idx = {'Low': 0, 'Medium': 1, 'High': 2}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.dataset.classes = ['Low', 'Medium', 'High']
        self.dataset.class_to_idx = self.class_to_idx

        # Nếu không include unlabeled thì loại bỏ ảnh trong thư mục 'unlabeled'
        valid_samples = []
        for path, _ in self.dataset.samples:
            label_name = os.path.basename(os.path.dirname(path))
            if include_unlabeled:
                # Trong trường hợp include_unlabeled = True, gán nhãn 'unlabeled' là -1 để phân biệt
                if label_name in self.class_to_idx:
                    valid_samples.append((path, self.class_to_idx[label_name]))
                elif label_name == 'unlabeled':
                    valid_samples.append((path, -1))  # Nhãn chưa biết
                else:
                    logging.warning(f"Found image in unknown folder '{label_name}' - ignoring: {path}")
            else:
                # Nếu không include unlabeled thì chỉ lấy ảnh có nhãn trong class_to_idx
                if label_name in self.class_to_idx:
                    valid_samples.append((path, self.class_to_idx[label_name]))
                else:
                    # Bỏ qua ảnh trong thư mục unlabeled hoặc bất kỳ thư mục lạ khác
                    pass
        self.dataset.samples = valid_samples

        self.transform = transform
        self.yolo_model = yolo_model
        self.root_dir = root_dir

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path, label = self.dataset.samples[idx]
        img_rgb, _ = self.dataset[idx]  # img_rgb: 3 x H x W
        
        # Nếu nhãn = -1 tức ảnh unlabeled thì vẫn return, nhưng label có thể -1
        if label == -1:
            label_value = -1
        else:
            label_value = label

        centroids = self.get_centroids(img_path, retry_with_lower_conf=False)
        if len(centroids) < 2:
            logging.warning(f"Không đủ centroids ({len(centroids)}) cho {img_path}. Sử dụng KDE mặc định.") 
            kde_channel = torch.zeros(1, IMAGE_SIZE, IMAGE_SIZE)
        else:
            try:
                x, y = zip(*centroids)
                x, y = np.array(x), np.array(y)
                # Kiểm tra độ lệch chuẩn để đảm bảo dữ liệu đủ đa dạng
                if np.std(x) < 1e-4 or np.std(y) < 1e-4:
                    logging.warning(f"Dữ liệu quá đồng nhất ({len(centroids)} centroids) cho {img_path}. Sử dụng KDE mặc định.")
                    kde_channel = torch.zeros(1, IMAGE_SIZE, IMAGE_SIZE)
                else:
                    # Thêm nhiễu lớn hơn
                    x += np.random.normal(0, 1e-4, len(x))
                    y += np.random.normal(0, 1e-4, len(y))
                    kde = gaussian_kde([x, y], bw_method=0.2)
                    X, Y = np.meshgrid(np.linspace(0, 1, IMAGE_SIZE), np.linspace(0, 1, IMAGE_SIZE))
                    positions = np.vstack([X.ravel(), Y.ravel()])
                    kde_values = kde(positions).reshape(IMAGE_SIZE, IMAGE_SIZE)
                    kde_channel = torch.tensor(kde_values, dtype=torch.float32).unsqueeze(0)
            except np.linalg.LinAlgError as e:
                logging.warning(f"Không thể tính KDE cho {img_path} ({len(centroids)} centroids): {e}. Sử dụng KDE mặc định.")
                kde_channel = torch.zeros(1, IMAGE_SIZE, IMAGE_SIZE)
        
        img_4ch = torch.cat([img_rgb, kde_channel], dim=0)  # 4 x H x W
        return img_4ch, label_value

    def get_centroids(self, img_path, retry_with_lower_conf=False):
        """Tính tọa độ tâm của các hộp giới hạn bằng YOLO."""
        if self.yolo_model is None:
            return []
        img = cv2.imread(img_path)
        if img is None:
            return []
        conf = YOLO_CONFIDENCE_THRESHOLD if not retry_with_lower_conf else 0.2
        results = self.yolo_model(img, verbose=False, conf=conf)
        centroids = []
        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) in VEHICLE_CLASS_IDS:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    centroid_x = (x1 + x2) / 2 / img.shape[1]
                    centroid_y = (y1 + y2) / 2 / img.shape[0]
                    centroids.append([centroid_x, centroid_y])
        # Thử lại với ngưỡng thấp hơn nếu không có centroids
        if len(centroids) == 0 and not retry_with_lower_conf:
            return self.get_centroids(img_path, retry_with_lower_conf=True)
        return centroids

# ----- CUSTOM RESNET50 (4 CHANNELS INPUT) -----
class ResNetWith4Channels(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

# ----- HÀM LOAD DỮ LIỆU -----
def load_data(include_unlabeled=False):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
    ])

    yolo_model = YOLO(YOLO_MODEL_NAME)
    train_dataset = TrafficDatasetWithKDE(os.path.join(DATA_DIR, 'train'), transform, yolo_model, include_unlabeled=include_unlabeled)
    valid_dataset = TrafficDatasetWithKDE(os.path.join(DATA_DIR, 'valid'), transform, yolo_model, include_unlabeled=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, valid_loader

# ----- HÀM KIỂM TRA DỮ LIỆU -----
def audit_dataset():
    """Quét tập dữ liệu để xác định hình ảnh có ít hoặc không có centroids."""
    model = YOLO(YOLO_MODEL_NAME)
    splits = ['train', 'valid']
    audit_results = {'Low': [], 'Medium': [], 'High': []}
    
    for split in splits:
        split_dir = os.path.join(DATA_DIR, split)
        for label in ['Low', 'Medium', 'High']:
            img_paths = glob.glob(os.path.join(split_dir, label, '*.jpg')) + glob.glob(os.path.join(split_dir, label, '*.png'))
            for img_path in img_paths:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                results = model(img, verbose=False, conf=YOLO_CONFIDENCE_THRESHOLD)
                vehicle_count = sum(1 for result in results for box in result.boxes if int(box.cls[0]) in VEHICLE_CLASS_IDS)
                if vehicle_count < 2:
                    audit_results[label].append((img_path, vehicle_count))
    
    # In kết quả kiểm tra
    print("Dataset audit results:")
    for label, issues in audit_results.items():
        if issues:
            print(f"Folder {label}:")
            for img_path, count in issues:
                print(f"  - {img_path}: {count} centroids")
    return audit_results

# ----- HÀM KIỂM TRA VÀ SỬA NHÃN DỮ LIỆU -----
def check_and_fix_labels():
    """Kiểm tra và sửa nhãn dữ liệu dựa trên số đếm phương tiện từ YOLO."""
    model = YOLO(YOLO_MODEL_NAME)
    splits = ['train', 'valid']
    for split in splits:
        split_dir = os.path.join(DATA_DIR, split)
        for label in ['Low', 'Medium', 'High']:
            os.makedirs(os.path.join(split_dir, label), exist_ok=True)

        for img_path in glob.glob(os.path.join(split_dir, '*', '*.jpg')) + glob.glob(os.path.join(split_dir, '*', '*.png')):
            img = cv2.imread(img_path)
            if img is None:
                continue
            results = model(img, verbose=False, conf=YOLO_CONFIDENCE_THRESHOLD)
            vehicle_count = sum(1 for result in results for box in result.boxes if int(box.cls[0]) in VEHICLE_CLASS_IDS)
            if vehicle_count <= COUNT_THRESHOLD_LOW_TO_MEDIUM:
                new_label = 'Low'
            elif vehicle_count <= COUNT_THRESHOLD_MEDIUM_TO_HIGH:
                new_label = 'Medium'
            else:
                new_label = 'High'
            new_path = os.path.join(split_dir, new_label, os.path.basename(img_path))
            if img_path != new_path:
                try:
                    shutil.move(img_path, new_path)
                except PermissionError as e:
                    print(f"Cannot move file {img_path}: {e}")

# ----- DỰ ĐOÁN NHÃN VÀ CHUYỂN ẢNH CHO UNLABELED -----
def predict_and_relabel_unlabeled(model):
    """Dùng model đã train để dự đoán nhãn cho ảnh trong thư mục 'unlabeled', sau đó chuyển ảnh vào thư mục tương ứng."""
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    unlabeled_dir = os.path.join(DATA_DIR, 'train', 'unlabeled')
    image_paths = glob.glob(os.path.join(unlabeled_dir, '*.jpg')) + glob.glob(os.path.join(unlabeled_dir, '*.png'))
    if len(image_paths) == 0:
        print("No images found in 'unlabeled' folder. Skipping relabeling.", flush=True)
        return

    print(f"Predicting labels for {len(image_paths)} images in 'unlabeled' folder...", flush=True)
    with torch.no_grad():
        for img_path in tqdm(image_paths):
            image = cv2.imread(img_path)
            if image is None:
                print(f"Cannot read image {img_path}", flush=True)
                continue
            # Chuyển đổi ảnh sang rgb và tensor, thêm channel KDE giả bằng 0 (1 channel 0)
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = transforms.ToPILImage()(img_rgb)
            img_tensor = transform(pil_img)  # 3 x H x W

            # Tạo channel KDE giả toàn 0 (1 x H x W)
            kde_channel = torch.zeros(1, IMAGE_SIZE, IMAGE_SIZE)
            img_4ch = torch.cat([img_tensor, kde_channel], dim=0)  # 4 x H x W

            img_4ch = img_4ch.unsqueeze(0).to(DEVICE)
            outputs = model(img_4ch)
            _, pred = outputs.max(1)
            pred_class = pred.item()
            class_name = ['Low', 'Medium', 'High'][pred_class]

            # Tạo thư mục nếu chưa tồn tại
            dest_dir = os.path.join(DATA_DIR, 'train', class_name)
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            dest_path = os.path.join(dest_dir, os.path.basename(img_path))
            try:
                shutil.move(img_path, dest_path)
                print(f"Moved {img_path} => {dest_path}")
            except Exception as e:
                print(f"Cannot move {img_path}: {e}")

# ----- TRAINING -----
def train_model():
    train_loader, valid_loader = load_data(include_unlabeled=False)  # Không train ảnh unlabeled
    model = ResNetWith4Channels(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_acc': [],
        'mse': [],
        'mae': [],
        'r2': [],
    }
    
    best_val_acc = 0
    early_stop_counter = 0
    PATIENCE = 5  # Số epoch không cải thiện trước khi dừng

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        print(f"\n===== Epoch {epoch+1}/{NUM_EPOCHS} =====", flush=True)
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            # Bỏ qua ảnh unlabeled
            mask = labels >= 0
            if mask.sum().item() == 0:
                continue
            images = images[mask]
            labels = labels[mask]

            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = correct / total if total > 0 else 0
        print(f"Train Loss: {total_loss:.4f} | Train Acc: {acc*100:.2f}%", flush=True)

        # Đánh giá trên tập validation
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            all_preds = []
            all_labels = []

            for images, labels in valid_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, preds = outputs.max(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            val_acc = correct / total if total > 0 else 0
            mse = mean_squared_error(all_labels, all_preds)
            mae = mean_absolute_error(all_labels, all_preds)
            r2 = r2_score(all_labels, all_preds)

            print(f"Validation Accuracy: {val_acc*100:.2f}% | MSE: {mse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}", flush=True)

            history['train_loss'].append(total_loss)
            history['train_acc'].append(acc)
            history['val_acc'].append(val_acc)
            history['mse'].append(mse)
            history['mae'].append(mae)
            history['r2'].append(r2)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                early_stop_counter = 0
                torch.save(model.state_dict(), "best_traffic_density_cnn.pth")  # Lưu model tốt nhất
                print("Best model saved.", flush=True)
            else:
                early_stop_counter += 1
                print(f"No improvement ({early_stop_counter}/{PATIENCE})", flush=True)
                if early_stop_counter >= PATIENCE:
                    print(f"Early stopping after {epoch+1} epochs.", flush=True)
                    break

        scheduler.step()

    # Lưu model
    torch.save(model.state_dict(), "traffic_density_cnn.pth")
    print("✅ Model saved to 'traffic_density_cnn.pth'")

    df_history = pd.DataFrame(history)
    df_history.to_csv("training_history.csv", index=False)
    print("Training history saved to 'training_history.csv'")

    # Dùng model dự đoán lại nhãn cho bộ ảnh unlabeled và chuyển ảnh
    predict_and_relabel_unlabeled(model)

if __name__ == "__main__":
    # Kiểm tra tập dữ liệu
    print("Auditing dataset...", flush=True)
    audit_results = audit_dataset()
    # Kiểm tra và sửa nhãn
    print("Checking and fixing labels...", flush=True)
    check_and_fix_labels()
    print("Start training...", flush=True)
    train_model()
