# 🚀 CLIP Prompt Ensemble Demo

Demo phân loại ảnh sử dụng **Prompt Engineering** và **Ensemble Methods** với mô hình CLIP.

## 📋 Giới thiệu

Dự án này minh họa cách sử dụng:
- **Prompt Engineering**: Tạo nhiều prompt templates khác nhau để tăng độ chính xác
- **Ensemble Methods**: Kết hợp kết quả từ nhiều prompts (Mean, Max, Weighted)
- **Zero-shot Learning**: Phân loại ảnh mà không cần training

### Đặc điểm nổi bật

✅ **10 Prompt Templates** - Đa dạng góc nhìn (blurry, bright, dark, close-up, ...)  
✅ **3 Ensemble Methods** - Mean, Max, Weighted aggregation  
✅ **10 Classes** - CIFAR-10 dataset (airplane, car, bird, cat, ...)  
✅ **Interactive Demo** - Test liên tục với random sampling  
✅ **Visualization** - Biểu đồ xác suất cho từng class  
✅ **Real-time Stats** - Theo dõi accuracy trong quá trình test  

---

## 🔧 Cài đặt

### 1. Yêu cầu hệ thống

- Python 3.7+
- RAM: 4GB+ (khuyến nghị 8GB)
- GPU: Không bắt buộc (có GPU sẽ nhanh hơn)

### 2. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

**Lưu ý**: Lần đầu chạy, CLIP sẽ tự động tải model (~350MB). Quá trình này có thể mất 2-5 phút tùy tốc độ mạng.

---

## 📁 Cấu trúc thư mục

```
clip_prompt_ensemble/
│
├── clip_prompt_ensemble.py    # File chính - Demo script
├── requirements.txt            # Dependencies
├── README.md                   # File này
│
└── images/                     # Thư mục chứa ảnh test
    ├── airplane/               # Ảnh máy bay
    ├── automobile/             # Ảnh ô tô
    ├── bird/                   # Ảnh chim
    ├── cat/                    # Ảnh mèo
    ├── deer/                   # Ảnh hươu
    ├── dog/                    # Ảnh chó
    ├── frog/                   # Ảnh ếch
    ├── horse/                  # Ảnh ngựa
    ├── ship/                   # Ảnh tàu
    └── truck/                  # Ảnh xe tải
```

### Chuẩn bị dữ liệu

Đặt ảnh test vào các thư mục tương ứng trong `images/`. Mỗi thư mục nên có **ít nhất 3-5 ảnh** để test đa dạng.

**Ví dụ**:
```
images/cat/cat1.jpg
images/cat/cat2.jpg
images/dog/dog1.png
images/airplane/plane1.jpg
```

**Định dạng hỗ trợ**: `.jpg`, `.jpeg`, `.png`

---

## 🎯 Cách sử dụng

### Chạy demo

```bash
python src/prompt_ensemble.py
```

### Quy trình sử dụng

1. **Chọn class**: Nhập số 1-10 để chọn class muốn test
   - Ví dụ: `3` → Bird (chim)
   - Nhấn Enter → Random chọn class

2. **Hệ thống tự động**:
   - Random chọn 1 ảnh từ class đã chọn
   - Phân loại ảnh bằng CLIP với 10 prompt templates
   - Hiển thị kết quả chi tiết
   - Lưu biểu đồ vào `classification_result.png`

3. **So sánh Ensemble Methods** (tùy chọn):
   - Nhập `y` → So sánh Mean, Max, Weighted
   - Nhập `n` → Bỏ qua

4. **Tiếp tục hoặc thoát**:
   - Nhấn Enter → Test class khác
   - Nhập `q` → Thoát chương trình

---

## 🔬 Các tính năng

### 1. Prompt Templates (10 mẫu)

```python
templates = [
    "a photo of a {}",              # Cơ bản
    "a blurry photo of a {}",       # Mờ
    "a bright photo of a {}",       # Sáng
    "a dark photo of a {}",         # Tối
    "a close-up photo of a {}",     # Cận cảnh
    "a photo of many {}",           # Nhiều đối tượng
    "a photo of the large {}",      # Kích thước lớn
    "a photo of the small {}",      # Kích thước nhỏ
    "a black and white photo of a {}",  # Đen trắng
    "a cropped photo of a {}",      # Cắt xén
]
```

### 2. Ensemble Methods

#### **Mean (Trung bình)**
```python
ensemble_scores = similarity.mean(dim=0)
```
- **Ưu điểm**: Cân bằng, ổn định
- **Phù hợp**: Hầu hết các trường hợp

#### **Max (Giá trị lớn nhất)**
```python
ensemble_scores = similarity.max(dim=0)[0]
```
- **Ưu điểm**: Nhạy với template phù hợp nhất
- **Phù hợp**: Khi có template rất "chắc chắn"

#### **Weighted (Trọng số)**
```python
weights = torch.linspace(0.5, 1.5, M).to(device)
ensemble_scores = (similarity * weights.unsqueeze(1)).sum(dim=0)
```
- **Ưu điểm**: Ưu tiên template quan trọng hơn
- **Phù hợp**: Khi biết template nào tốt hơn

### 3. Thống kê real-time

Chương trình tự động theo dõi:
- Tổng số test đã thực hiện
- Số lượng dự đoán đúng/sai
- **Accuracy** (%) tổng thể

---

## 📊 Output

### 1. Console Output

```
==========================================================
PHÂN LOẠI ẢNH: images/cat/cat1.jpg
==========================================================
✓ Image loaded: (640, 480)
✓ Preprocessed: torch.Size([1, 3, 224, 224])

✓ Image features: torch.Size([1, 512])
✓ Normalized: ||v|| = 1.0000

✓ Similarity matrix computed: torch.Size([10, 10])

Similarity Matrix (Top-3 classes for first 3 templates):
  Template 1: cat=24.5  dog=22.3  deer=18.1  
  Template 2: cat=23.8  dog=21.9  horse=17.5  
  Template 3: cat=25.2  dog=22.7  deer=18.8  

✓ Ensemble method: MEAN (Trung bình cộng)

==========================================================
KẾT QUẢ PHÂN LOẠI
==========================================================
Predicted Class: CAT
Confidence: 78.45%

Top-5 Predictions:
  1. cat         : 78.45% ███████████████████████████████████████
  2. dog         : 12.34% ██████
  3. deer        :  4.23% ██
  4. horse       :  2.67% █
  5. bird        :  1.12% 
```

### 2. Visualization (classification_result.png)

Biểu đồ gồm 2 phần:
- **Trái**: Ảnh gốc đầu vào
- **Phải**: Bar chart xác suất cho 10 classes

---

## ⚙️ Cấu hình

### Thay đổi số lượng templates

Mở `clip_prompt_ensemble.py`, tìm biến `templates` và thêm/bớt:

```python
templates = [
    "a photo of a {}",
    "a rendering of a {}",      # Thêm template mới
    "a cropped photo of a {}",
    # ... thêm nhiều templates khác
]
```

### Thay đổi classes

Thay đổi biến `class_names`:

```python
class_names = [
    'airplane', 'automobile', 'bird',  # Giữ nguyên
    'lion', 'tiger',                   # Thêm classes mới
]
```

**Lưu ý**: Phải tạo thư mục tương ứng trong `images/`

### Chọn device (GPU/CPU)

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

Để **bắt buộc dùng CPU**:
```python
device = "cpu"
```

---

## 🎓 Kiến thức nền tảng

### 1. CLIP Model

- **Paper**: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- **Kiến trúc**: Vision Transformer (ViT-B/32)
- **Huấn luyện**: 400M image-text pairs từ internet
- **Zero-shot**: Không cần training, dự đoán trực tiếp

### 2. Prompt Engineering

Thay vì dùng 1 prompt cố định, ta dùng nhiều prompts đa dạng:
- Tăng **robustness** (ổn định)
- Bao quát nhiều **variations** (biến thể)
- Giảm **bias** (thiên lệch) của 1 prompt đơn lẻ

### 3. Ensemble Learning

Kết hợp nhiều "weak learners" → 1 "strong learner":
- **Giảm variance** - Ổn định hơn
- **Tăng accuracy** - Chính xác hơn
- **Robust** - Ít bị nhiễu hơn

---

## 📈 Kết quả mong đợi

Với dataset CIFAR-10:

| Ensemble Method | Accuracy (Expected) |
|----------------|---------------------|
| Single Prompt  | ~75-80%            |
| Mean Ensemble  | ~82-87%            |
| Max Ensemble   | ~80-85%            |
| Weighted       | ~83-88%            |

**Lưu ý**: Kết quả thực tế phụ thuộc vào:
- Chất lượng ảnh test
- Độ tương đồng với dữ liệu training của CLIP
- Số lượng templates sử dụng

---

## 🐛 Troubleshooting

### 1. Lỗi "No module named 'clip'"

```bash
pip install git+https://github.com/openai/CLIP.git
```

### 2. Lỗi "CUDA out of memory"

Chuyển sang CPU:
```python
device = "cpu"
```

### 3. Model tải chậm

- Kiểm tra kết nối internet
- Model (~350MB) chỉ tải 1 lần duy nhất
- Lần sau sẽ load từ cache

### 4. Không có ảnh trong thư mục

```
⚠ CẢNH BÁO: Không có ảnh cho class 'cat'
Vui lòng đặt ảnh vào folder: images/cat
```

→ Tạo thư mục và thêm ảnh

### 5. Accuracy thấp

- Kiểm tra chất lượng ảnh (rõ ràng, đúng class)
- Thử tăng số lượng templates
- Thử các ensemble methods khác

---

## 🔮 Mở rộng

### 1. Thêm Few-shot Learning

Kết hợp với `few_shot.py` (nếu có) để training thêm với ít ảnh

### 2. Custom Templates

Thiết kế templates phù hợp với domain cụ thể:
```python
# Ví dụ: Medical domain
templates = [
    "an X-ray image of a {}",
    "a CT scan showing {}",
    "a medical image of {}",
]
```

### 3. Batch Processing

Xử lý nhiều ảnh cùng lúc:
```python
for image_path in image_list:
    classify_image(image_path, show_details=False)
```

### 4. Web API

Tạo Flask/FastAPI endpoint để deploy lên server

---

## 📚 Tài liệu tham khảo

- [CLIP Paper (OpenAI)](https://arxiv.org/abs/2103.00020)
- [CLIP GitHub Repository](https://github.com/openai/CLIP)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [Ensemble Learning Overview](https://en.wikipedia.org/wiki/Ensemble_learning)

---

## 🙏 Acknowledgments

- OpenAI team cho CLIP model
- PyTorch team
- Cộng đồng Computer Vision

---

**Happy Coding! 🚀**