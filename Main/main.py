# --- IMPORT THƯ VIỆN ---

import os
import glob
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import shutil
import random
import folium
import concurrent.futures
from PIL import Image, ImageTk
import threading
import tkinter.ttk as ttk
import torch
import torch.nn as nn
from torchvision import transforms
import datetime
import subprocess

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
except ImportError:
    tk = filedialog = None
    print("Cảnh báo: Thư viện tkinter không khả dụng.")

# --- CẤU HÌNH ---
YOLO_MODEL_NAME = 'yolov8n.pt'
VEHICLE_CLASS_IDS_TO_COUNT = [2, 3, 5, 7]  # COCO: 2:car, 3:motorcycle, 5:bus, 7:truck
YOLO_CONFIDENCE_THRESHOLD = 0.3
KDE_BANDWIDTH = 30
DOWNSCALE_FACTOR_FOR_KDE = 4
COUNT_THRESHOLD_LOW_TO_MEDIUM = 5
COUNT_THRESHOLD_MEDIUM_TO_HIGH = 15

# --- MODEL DENSITY CNN ---
class ResNetWith4Channels(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        from torchvision import models
        self.resnet = models.resnet50(weights=None)
        self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    def forward(self, x):
        return self.resnet(x)

density_cnn_model = ResNetWith4Channels(num_classes=3)
try:
    density_cnn_model.load_state_dict(torch.load("best_traffic_density_cnn.pth", map_location="cpu"))
    density_cnn_model.eval()
except Exception as e:
    print("Warning: Could not load best_traffic_density_cnn.pth:", e)

def count_motorbikes(yolo_results):
    count = 0
    for yolo_result in yolo_results:
        for box in yolo_result.boxes:
            if int(box.cls[0]) == 3:  # 3 là motorcycle trong COCO
                count += 1
    return count

def predict_density_cnn(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img)
    kde_channel = torch.zeros(1, 224, 224)
    input_tensor = torch.cat([img_tensor, kde_channel], dim=0).unsqueeze(0)
    with torch.no_grad():
        output = density_cnn_model(input_tensor)
        pred_class = output.argmax(dim=1).item()
    if pred_class == 0:
        return 3
    elif pred_class == 1:
        return 10
    else:
        return 20

# --- HÀM XỬ LÝ ẢNH ---
def simple_preprocess_image(image_path, resize_max_dim=640):
    img = cv2.imread(image_path)
    if img is not None and max(img.shape[:2]) > resize_max_dim:
        scale = resize_max_dim / max(img.shape[:2])
        img = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))
    return img

def count_vehicles_and_get_results(image_np, model, target_classes, confidence_thresh):
    vehicle_count = 0
    vehicle_centers = []
    if image_np is None or image_np.size == 0: return 0, [], None
    try:
        results = model(image_np, verbose=False, conf=confidence_thresh)
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                if class_id in target_classes:
                    vehicle_count += 1
                    x1, y1, x2, y2 = box.xyxy[0]
                    vehicle_centers.append((int((x1 + x2) / 2), int((y1 + y2) / 2)))
        return vehicle_count, vehicle_centers, results
    except Exception as e:
        print(f"Lỗi nhỏ khi chạy YOLO: {e}")
        return 0, [], None

def generate_kde_map(image_shape, centers, bandwidth, downscale_factor):
    if not centers or len(centers) < 2:
        output_shape = (image_shape[0] // downscale_factor, image_shape[1] // downscale_factor)
        if output_shape[0] <=0 or output_shape[1] <= 0: output_shape = (max(1, image_shape[0]//downscale_factor), max(1, image_shape[1]//downscale_factor))
        return np.zeros(output_shape, dtype=np.float32)
    centers_array = np.array(centers)
    x_coords = centers_array[:, 0] / downscale_factor
    y_coords = centers_array[:, 1] / downscale_factor
    grid_y_max = image_shape[0] // downscale_factor
    grid_x_max = image_shape[1] // downscale_factor
    if grid_x_max <= 0 or grid_y_max <=0: return np.zeros((max(1,grid_y_max), max(1,grid_x_max)), dtype=np.float32)
    xx, yy = np.mgrid[0:grid_x_max:1, 0:grid_y_max:1]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    try:
        if len(np.unique(x_coords)) < 2 or len(np.unique(y_coords)) < 2 or centers_array.shape[0] < min(centers_array.shape[1]+1, 2):
             return np.zeros((grid_y_max, grid_x_max), dtype=np.float32)
        effective_bandwidth = bandwidth / downscale_factor
        kernel = stats.gaussian_kde(np.vstack([x_coords, y_coords]), bw_method=effective_bandwidth / np.mean([grid_x_max, grid_x_max]))
        density_map_flat = kernel(positions)
        density_map = np.reshape(density_map_flat.T, xx.shape)
        return density_map.T
    except Exception as e: return np.zeros((grid_y_max, grid_x_max), dtype=np.float32)

def assign_density_label(vehicle_count):
    count = int(vehicle_count)
    if count <= COUNT_THRESHOLD_LOW_TO_MEDIUM:
        return 'Low'
    elif count > COUNT_THRESHOLD_LOW_TO_MEDIUM and count <= COUNT_THRESHOLD_MEDIUM_TO_HIGH:
        return 'Medium'
    else:
        return 'High'

def visualize_on_map(df_or_csv, output_html='traffic_alert_map.html'):
    if isinstance(df_or_csv, str):
        df = pd.read_csv(df_or_csv)
    else:
        df = df_or_csv
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        print("CSV chưa có cột latitude/longitude. Không thể tạo bản đồ.")
        return
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
    color_map = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=8,
            color=color_map.get(row['density_label'], 'blue'),
            fill=True,
            fill_color=color_map.get(row['density_label'], 'blue'),
            fill_opacity=0.7,
            popup=f"{row['filename']}<br>Mật độ: {row['density_label']}<br>Số xe: {row['vehicle_count']}"
        ).add_to(m)
    m.save(output_html)
    print(f"Đã lưu bản đồ cảnh báo tại: {output_html}")

def load_street_locations_csv(csv_path):
    df = pd.read_csv(csv_path)
    loc_dict = {row['filename']: (row['latitude'], row['longitude']) for _, row in df.iterrows()}
    return loc_dict

# --- GIAO DIỆN CHÍNH ---
class TrafficApp:
    def __init__(self, master):
        self.master = master
        master.title("🚦 Traffic Density Analyzer")
        master.geometry("1600x900")
        self.image_paths = []
        self.results = []
        self.yolo_model = YOLO(YOLO_MODEL_NAME)
        self.images_per_page = 16
        self.current_page = 0
        self.locations_dict = {}
        self.setup_layout()

    def auto_train(self):
        progress = tk.Toplevel(self.master)
        progress.title("Đang huấn luyện mô hình CNN...")
        tk.Label(progress, text="Đang tự động huấn luyện mô hình CNN, vui lòng chờ...", font=("Segoe UI", 12)).pack(padx=20, pady=10)
        log_text = tk.Text(progress, height=20, width=80, font=("Consolas", 10))
        log_text.pack(padx=20, pady=10)
        pb = ttk.Progressbar(progress, orient="horizontal", length=400, mode="indeterminate")
        pb.pack(padx=20, pady=10)
        pb.start()
        progress.update()
        def run_train():
            import sys
            log_file = open("./logs/train_log.txt", "w", encoding="utf-8")
            try:
                proc = subprocess.Popen(
                    [sys.executable, "../Train/train_density_cnn.py"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
                for line in proc.stdout:
                    log_file.write(line)
                    log_file.flush()
                    if line.startswith("PROGRESS:"):
                        try:
                            cur, total = map(int, line.strip().split()[1].split("/"))
                            pb.config(mode="determinate", maximum=total)
                            pb["value"] = cur
                        except Exception:
                            pass
                    else:
                        log_text.insert(tk.END, line)
                        log_text.see(tk.END)
                proc.wait()
                if proc.returncode == 0:
                    messagebox.showinfo("Thành công", "Đã huấn luyện xong mô hình CNN!")
                else:
                    messagebox.showerror("Lỗi", "Huấn luyện thất bại. Xem log để biết chi tiết.")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Lỗi khi huấn luyện mô hình:\n{e}")
            pb.stop()
            progress.destroy()
        threading.Thread(target=run_train).start()

    def setup_layout(self):
        # Sidebar style
        self.sidebar = tk.Frame(self.master, bg="#e3eaf2", width=320)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.result_area = tk.Frame(self.master, bg="white")
        self.result_area.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Tiêu đề phần mềm
        tk.Label(self.sidebar, text="🚦 TRAFFIC DENSITY ANALYZER", font=("Segoe UI", 18, "bold"), bg="#e3eaf2", fg="#2a3f5f").pack(pady=(30, 20))

        # Nút Auto Train
        # self.btn_auto_train = tk.Button(self.sidebar, text="🚀 Huấn Luyện Mô Hình CNN", font=("Segoe UI", 14, "bold"), bg="#4caf50", fg="white", height=2, command=self.auto_train, relief=tk.RAISED, bd=2)
        # self.btn_auto_train.pack(fill=tk.X, padx=30, pady=(0, 20))

        self.btn_select_folder = tk.Button(self.sidebar, text="📂 Chọn thư mục ảnh", font=("Segoe UI", 13), bg="#ffffff", fg="#2a3f5f", height=2, command=self.select_folder, relief=tk.GROOVE, bd=2)
        self.btn_select_folder.pack(fill=tk.X, padx=30, pady=8)

        self.btn_select_location = tk.Button(self.sidebar, text="🗺️ Chọn file vị trí GPS", font=("Segoe UI", 13), bg="#ffffff", fg="#2a3f5f", height=2, command=self.select_location_file, relief=tk.GROOVE, bd=2)
        self.btn_select_location.pack(fill=tk.X, padx=30, pady=8)

        self.btn_analyze = tk.Button(self.sidebar, text="🔎 Phân tích", font=("Segoe UI", 13, "bold"), bg="#1976d2", fg="white", height=2, command=self.analyze_all, state='disabled', relief=tk.RAISED, bd=2)
        self.btn_analyze.pack(fill=tk.X, padx=30, pady=8)

        self.btn_export_map = tk.Button(self.sidebar, text="🌐 Xuất bản đồ", font=("Segoe UI", 13), bg="#ffffff", fg="#2a3f5f", height=2, command=self.export_map, state='disabled', relief=tk.GROOVE, bd=2)
        self.btn_export_map.pack(fill=tk.X, padx=30, pady=8)

        self.btn_show_stats = tk.Button(self.sidebar, text="📊 Biểu đồ thống kê", font=("Segoe UI", 13), bg="#ffffff", fg="#2a3f5f", height=2, command=self.show_stats, relief=tk.GROOVE, bd=2)
        self.btn_show_stats.pack(fill=tk.X, padx=30, pady=8)

        self.btn_reset = tk.Button(self.sidebar, text="🔄 Làm mới", font=("Segoe UI", 13), bg="#f44336", fg="white", height=2, command=self.reset, relief=tk.RAISED, bd=2)
        self.btn_reset.pack(fill=tk.X, padx=30, pady=(30, 8))

        # Navigation
        nav_frame = tk.Frame(self.sidebar, bg="#e3eaf2")
        nav_frame.pack(fill=tk.X, padx=30, pady=(40, 10))
        self.btn_prev_page = tk.Button(nav_frame, text="⬅️ Trang trước", font=("Segoe UI", 12), command=self.prev_page, state='disabled')
        self.btn_prev_page.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
        self.page_label = tk.Label(nav_frame, text="Trang 1", font=("Segoe UI", 12), bg="#e3eaf2")
        self.page_label.pack(side=tk.LEFT, padx=5)
        self.btn_next_page = tk.Button(nav_frame, text="Trang sau ➡️", font=("Segoe UI", 12), command=self.next_page, state='disabled')
        self.btn_next_page.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(5, 0))

        # Result grid area
        self.result_canvas = tk.Canvas(self.result_area, bg="white")
        self.result_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.result_frame = tk.Frame(self.result_canvas, bg="white")
        self.result_canvas.create_window((0, 0), window=self.result_frame, anchor='nw')
        self.result_frame.bind("<Configure>", lambda e: self.result_canvas.configure(scrollregion=self.result_canvas.bbox("all")))
        self.v_scroll = tk.Scrollbar(self.result_area, orient=tk.VERTICAL, command=self.result_canvas.yview)
        self.result_canvas.configure(yscrollcommand=self.v_scroll.set)
        self.v_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def select_folder(self):
        folder = filedialog.askdirectory(title="Chọn thư mục ảnh")
        if folder:
            exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff')
            self.image_paths = []
            for ext in exts:
                self.image_paths.extend(glob.glob(os.path.join(folder, ext)))
            self.results = [None] * len(self.image_paths)
            self.current_page = 0
            self.btn_analyze.config(state='normal')
            self.btn_export_map.config(state='disabled')
            self.show_page()
            self.update_nav_buttons()

    def select_location_file(self):
        path = filedialog.askopenfilename(title="Chọn file vị trí GPS", filetypes=[("CSV files", "*.csv")])
        if path:
            self.locations_dict = load_street_locations_csv(path)
            messagebox.showinfo("Thành công", "Đã nạp file vị trí GPS!")

    def reset(self):
        self.image_paths = []
        self.results = []
        self.current_page = 0
        self.clear_result_frame()
        self.btn_analyze.config(state='disabled')
        self.btn_export_map.config(state='disabled')
        self.update_nav_buttons()

    def analyze_all(self):
        if not self.image_paths:
            return
        if not self.locations_dict:
            messagebox.showwarning("Thiếu vị trí GPS", "Bạn phải chọn file vị trí GPS trước khi phân tích!")
            return
        self.btn_analyze.config(state='disabled')
        self.clear_result_frame()
        progress = tk.Toplevel(self.master)
        progress.title("Đang phân tích...")
        tk.Label(progress, text="Đang phân tích ảnh, vui lòng chờ...", font=("Segoe UI", 12)).pack(padx=20, pady=10)
        pb = ttk.Progressbar(progress, orient="horizontal", length=400, mode="determinate", maximum=len(self.image_paths))
        pb.pack(padx=20, pady=10)
        progress.update()
        def batch_analysis():
            for idx, img_path in enumerate(self.image_paths):
                result = self.traffic_image_pipeline(img_path)
                # Đếm xe máy bằng density_cnn
                motorbike_count = predict_density_cnn(img_path)
                result["motorbike_count"] = motorbike_count
                result["vehicle_count"] += motorbike_count
                fname = os.path.basename(img_path)
                if fname in self.locations_dict:
                    lat, lon = self.locations_dict[fname]
                    result["latitude"] = lat
                    result["longitude"] = lon
                self.results[idx] = result
                pb["value"] = idx + 1
                progress.update()
            progress.destroy()
            self.show_page()
            self.update_nav_buttons()
            self.btn_export_map.config(state='normal')
            messagebox.showinfo("Hoàn thành", "Đã phân tích xong tất cả ảnh!")
        threading.Thread(target=batch_analysis).start()

    def traffic_image_pipeline(self, image_path):
        img_np = simple_preprocess_image(image_path)
        if img_np is None:
            return {
                "success": False,
                "error": f"Không đọc được ảnh: {image_path}"
            }
        vehicle_count, vehicle_centers, yolo_results = count_vehicles_and_get_results(
            img_np, self.yolo_model, VEHICLE_CLASS_IDS_TO_COUNT, YOLO_CONFIDENCE_THRESHOLD
        )
        kde_map = None
        kde_max_density = 0.0
        if vehicle_centers:
            kde_map = generate_kde_map(img_np.shape, vehicle_centers, KDE_BANDWIDTH, DOWNSCALE_FACTOR_FOR_KDE)
            if kde_map is not None and kde_map.size > 0:
                kde_max_density = np.max(kde_map)
        density_label = assign_density_label(vehicle_count)
        return {
            "success": True,
            "image_path": image_path,
            "vehicle_count": vehicle_count,
            "vehicle_centers": vehicle_centers,
            "density_label": density_label,
            "kde_max_density": kde_max_density,
            "yolo_results": yolo_results,
            "kde_map": kde_map
        }

    def export_map(self):
        data = []
        for result in self.results:
            if result and result.get("success", True):
                lat = result.get("latitude", None)
                lon = result.get("longitude", None)
                if lat is not None and lon is not None:
                    data.append({
                        "filename": os.path.basename(result["image_path"]),
                        "latitude": lat,
                        "longitude": lon,
                        "density_label": result["density_label"],
                        "vehicle_count": result["vehicle_count"]
                    })
        if not data:
            messagebox.showwarning("Thiếu dữ liệu", "Không có dữ liệu vị trí để xuất bản đồ!")
            return
        df = pd.DataFrame(data)
        now = datetime.datetime.now()
        time_str = now.strftime("Report_%d%m%Y_%Ih%Mm_%p.html")
        output_html = filedialog.asksaveasfilename(
            defaultextension=".html",
            filetypes=[("HTML files", "*.html")],
            title="Lưu bản đồ cảnh báo",
            initialfile=time_str
        )
        if output_html:
            visualize_on_map(df, output_html)
            messagebox.showinfo("Thành công", f"Đã lưu bản đồ cảnh báo tại:\n{output_html}")

    def show_page(self):
        self.clear_result_frame()
        start = self.current_page * self.images_per_page
        end = min(start + self.images_per_page, len(self.image_paths))
        for i, idx in enumerate(range(start, end)):
            row, col = divmod(i, 4)
            self.show_image_result(idx, row, col)
        total_pages = max(1, (len(self.image_paths) + self.images_per_page - 1) // self.images_per_page)
        self.page_label.config(text=f"Trang {self.current_page + 1} / {total_pages}")

    def show_image_result(self, idx, row, col):
        img_path = self.image_paths[idx]
        result = self.results[idx]
        cell = tk.Frame(self.result_frame, bg="white", relief=tk.RIDGE, borderwidth=1)
        cell.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
        img = cv2.imread(img_path)
        if img is not None and result and result["yolo_results"]:
            img_display = img.copy()
            count_on_image = 0
            for yolo_result in result["yolo_results"]:
                for box in yolo_result.boxes:
                    class_id = int(box.cls[0])
                    if class_id in VEHICLE_CLASS_IDS_TO_COUNT:
                        x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
                        cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # Đánh số thứ tự cho từng xe
                        # text_label = str(count_on_image + 1)
                        # (text_width, text_height), baseline = cv2.getTextSize(text_label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                        # text_origin = (x1, max(y1 - 10, text_height + 5))
                        # cv2.putText(img_display, text_label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                        # count_on_image += 1
            img = img_display
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)
            img_pil.thumbnail((350, 250))
            tk_img = ImageTk.PhotoImage(img_pil)
            img_label = tk.Label(cell, image=tk_img, bg="white", cursor="hand2")
            img_label.image = tk_img
            img_label.pack(pady=2)
            img_label.bind("<Button-1>", lambda e, path=img_path, result=result: self.show_zoom_image(path, result))
        else:
            img_label = tk.Label(cell, text="Không thể đọc ảnh", bg="white")
            img_label.pack(pady=2)
        info = ""
        info += f"Tên file: {os.path.basename(img_path)}\n"
        if result:
            info += f"Số lượng phương tiện: {result['vehicle_count']}\n"
            info += f"Mật độ: {result['density_label']}\n"
            info += f"KDE lớn nhất: {result['kde_max_density']:.4f}\n"
        else:
            info += "Chưa phân tích\n"
        tk.Label(cell, text=info, justify="left", anchor="w", bg="white", font=("Segoe UI", 10)).pack()

    def show_zoom_image(self, img_path, result):
        top = tk.Toplevel(self.master)
        top.title(f"Xem chi tiết: {os.path.basename(img_path)}")
        img = cv2.imread(img_path)
        if img is not None and result and result["yolo_results"]:
            img_display = img.copy()
            count_on_image = 0
            for yolo_result in result["yolo_results"]:
                for box in yolo_result.boxes:
                    class_id = int(box.cls[0])
                    if class_id in VEHICLE_CLASS_IDS_TO_COUNT:
                        x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
                        cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # Đếm số thứ tự khi zoom
                        # text_label = str(count_on_image + 1)
                        # (text_width, text_height), baseline = cv2.getTextSize(text_label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
                        # text_origin = (x1, max(y1 - 10, text_height + 5))
                        # cv2.putText(img_display, text_label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
                        # count_on_image += 1
            img = img_display
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)
            img_pil.thumbnail((900, 700))
            tk_img = ImageTk.PhotoImage(img_pil)
            label = tk.Label(top, image=tk_img)
            label.image = tk_img
            label.pack()
        else:
            tk.Label(top, text="Không thể đọc ảnh").pack()

    def clear_result_frame(self):
        for widget in self.result_frame.winfo_children():
            widget.destroy()

    def prev_page(self):
        if self.current_page > 0:
            self.current_page -= 1
            self.show_page()
            self.update_nav_buttons()

    def next_page(self):
        total_pages = max(1, (len(self.image_paths) + self.images_per_page - 1) // self.images_per_page)
        if self.current_page < total_pages - 1:
            self.current_page += 1
            self.show_page()
            self.update_nav_buttons()

    def update_nav_buttons(self):
        total_pages = max(1, (len(self.image_paths) + self.images_per_page - 1) // self.images_per_page)
        self.btn_prev_page.config(state='normal' if self.current_page > 0 else 'disabled')
        self.btn_next_page.config(state='normal' if self.current_page < total_pages - 1 else 'disabled')

    def show_stats(self):
        counts = []
        labels = []
        for result in self.results:
            if result and result.get("success", True):
                counts.append(result["vehicle_count"])
                labels.append(result["density_label"])
        if not counts:
            messagebox.showwarning("Chưa có dữ liệu", "Chưa có kết quả phân tích để thống kê!")
            return
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.hist(counts, bins=10, color='skyblue', edgecolor='black')
        plt.title("Phân phối số lượng phương tiện")
        plt.xlabel("Số lượng phương tiện")
        plt.ylabel("Số ảnh")
        plt.subplot(1,2,2)
        pd.Series(labels).value_counts().plot(kind='bar', color=['green','orange','red'])
        plt.title("Thống kê nhãn mật độ")
        plt.xlabel("Nhãn mật độ")
        plt.ylabel("Số ảnh")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficApp(root)
    root.mainloop()