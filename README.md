# TrafficDensityAnalyzer

Ứng dụng phân tích mật độ giao thông bằng công nghệ YOLOv8 và CNN.

## Mục đích
Dự án này giúp phân tích ảnh giao thông, đếm số lượng phương tiện, đánh giá nhãn mật độ (`Low`, `Medium`, `High`) và tạo bản đồ cảnh báo vị trí.

## Tính năng chính
- Giao diện GUI bằng `tkinter` để chọn thư mục ảnh và file GPS.
- Dùng `ultralytics YOLOv8` để phát hiện phương tiện (xe hơi, xe máy, xe buýt, xe tải).
- Tạo bản đồ mật độ KDE từ các tâm hộp giới hạn.
- Huấn luyện và dùng mô hình `ResNet50` đầu vào 4 kênh (RGB + kênh KDE) để dự đoán mật độ.
- Xuất báo cáo HTML bản đồ cảnh báo với vị trí GPS.
- Hiển thị thống kê phân phối số lượng phương tiện và nhãn mật độ.

## Yêu cầu

- Python 3.8/3.9/3.10
- PyTorch
- torchvision
- ultralytics
- opencv-python
- numpy
- pandas
- scipy
- matplotlib
- seaborn
- pillow
- folium
- tqdm

## Cài đặt nhanh

```bash
pip install torch torchvision ultralytics opencv-python numpy pandas scipy matplotlib seaborn pillow folium tqdm
```

## Chạy ứng dụng

1. Mở terminal vào thư mục gốc repo.
2. Chạy:

```bash
python Main/main.py
```

3. Trong ứng dụng:
- Chọn thư mục chứa ảnh.
- Chọn file `locations.csv` chứa thông tin GPS.
- Nhấn `Phân tích` để bắt đầu xử lý.
- Sau khi hoàn thành, nhấn `Xuất bản đồ` để lưu báo cáo HTML.

## Huấn luyện mô hình CNN

Tập lệnh huấn luyện nằm ở `Train/train_density_cnn.py`.

```bash
python Train/train_density_cnn.py
```

Nó sử dụng dữ liệu đã xử lý trong `Train/traffic_analysis_output/processed_for_cnn/` để huấn luyện mô hình phát hiện mật độ.

## Dữ liệu và mô hình

- `yolov8n.pt`: mô hình YOLOv8 để phát hiện phương tiện.
- `best_traffic_density_cnn.pth`: mô hình ResNet50 đã huấn luyện để đánh giá mật độ.
- `locations.csv`: file vị trí GPS tương ứng với tên ảnh.

## Lưu ý

- Thư mục `.venv` không nên đưa vào git; đã được loại trừ bằng `.gitignore`.
- Nếu muốn phân tích ảnh mới, hãy chuẩn bị ảnh và file vị trí GPS đúng tên file.
- Dữ liệu huấn luyện và ảnh lớn có thể khiến repo nặng.
