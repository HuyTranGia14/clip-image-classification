# Zero-Shot Image Classification with CLIP

Dự án phân loại ảnh zero-shot sử dụng mô hình CLIP (Contrastive Language-Image Pre-training) của OpenAI để phân loại ảnh thuộc **10 classes của CIFAR-10** mà không cần huấn luyện lại mô hình.

## Mô tả

Dự án này sử dụng CLIP ViT-B/32 để thực hiện phân loại ảnh zero-shot. CLIP là mô hình đa phương thức (multimodal) được huấn luyện trên hàng triệu cặp ảnh-văn bản, cho phép phân loại ảnh dựa trên mô tả văn bản mà không cần huấn luyện thêm trên tập dữ liệu mục tiêu.

**Đặc điểm:**
- ✅ **Chung dataset với Few-shot và Prompt Ensemble**: Sử dụng cùng 50 ảnh (5 ảnh/class × 10 classes) để đảm bảo tính đồng bộ khi so sánh kết quả
- ✅ **10 classes CIFAR-10**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- ✅ **10 ảnh ví dụ**: Hiển thị và lưu kết quả dự đoán cho 10 ảnh đầu tiên

## Cấu trúc thư mục

```
clip-image-classification/
├── zero_shot.py                   # File chính chứa code
├── requirements_zero_shot.txt     # Các thư viện cần thiết
├── README_ZERO_SHOT.md           # File hướng dẫn này
└── images/                        # Thư mục chứa dữ liệu (chung với few-shot & prompt ensemble)
    ├── airplane/                  # 5 ảnh
    ├── automobile/                # 5 ảnh
    ├── bird/                      # 5 ảnh
    ├── cat/                       # 5 ảnh
    ├── deer/                      # 5 ảnh
    ├── dog/                       # 5 ảnh
    ├── frog/                      # 5 ảnh
    ├── horse/                     # 5 ảnh
    ├── ship/                      # 5 ảnh
    └── truck/                     # 5 ảnh
```

## Yêu cầu hệ thống

- Python 3.7+
- CUDA-compatible GPU (khuyến nghị, có thể chạy trên CPU nhưng chậm hơn)

## Cài đặt

1. Clone hoặc tải về repository này

2. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

Danh sách thư viện:
- `torch` - PyTorch framework
- `torchvision` - Xử lý ảnh và dataset
- `numpy` - Tính toán số học
- `pillow` - Xử lý ảnh
- `scikit-learn` - Tính confusion matrix
- `matplotlib` - Vẽ biểu đồ
- `clip` - Mô hình CLIP của OpenAI

## Chuẩn bị dữ liệu

Đặt các ảnh vào thư mục `images/` với cấu trúc:
```
images/
├── airplane/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ... (5 ảnh)
├── automobile/
│   └── ... (5 ảnh)
├── bird/
│   └── ... (5 ảnh)
├── cat/
│   └── ... (5 ảnh)
├── deer/
│   └── ... (5 ảnh)
├── dog/
│   └── ... (5 ảnh)
├── frog/
│   └── ... (5 ảnh)
├── horse/
│   └── ... (5 ảnh)
├── ship/
│   └── ... (5 ảnh)
└── truck/
    └── ... (5 ảnh)
```

Mỗi thư mục con đại diện cho một lớp (class) và chứa các ảnh thuộc lớp đó. Đây là **dataset chung** được sử dụng bởi cả 3 phương pháp (Zero-shot, Few-shot, Prompt Ensemble) để đảm bảo sự đồng bộ khi so sánh kết quả.

## Chạy chương trình

```bash
python src/zero_shot.py
```

## Kết quả đầu ra

Khi chạy xong, chương trình sẽ:

1. **In ra console:**
   - Thông tin thiết bị (CPU/GPU)
   - Số lượng ảnh trong dataset
   - Danh sách các prompts được tạo
   - Zero-shot Top-1 Accuracy (độ chính xác tổng thể)
   - Confusion matrix dạng số (10×10)
   - Độ chính xác theo từng lớp (10 classes)
   - Ví dụ dự đoán cho 10 ảnh đầu tiên

2. **Tạo các file:**
   - `confusion_matrix.png` - Ma trận nhầm lẫn 10×10 (confusion matrix) trực quan
   - `example_XX_<true_label>_pred_<predicted_label>.png` - 10 ảnh ví dụ với nhãn thực tế và dự đoán

## Cách hoạt động

1. **Load mô hình CLIP:** Tải mô hình ViT-B/32 đã được pre-train
2. **Tạo text features:** Chuyển đổi các prompts (ví dụ: "a photo of a cat") thành vector embeddings
3. **Tạo image features:** Chuyển đổi mỗi ảnh thành vector embeddings
4. **Tính similarity:** So sánh độ tương đồng giữa image features và text features
5. **Dự đoán:** Chọn lớp có độ tương đồng cao nhất

## Tùy chỉnh

### Thay đổi template prompt

Trong hàm `build_text_features()`, bạn có thể thay đổi template:
```python
template = "a photo of a {}"  # Mặc định
# Hoặc thử:
template = "an image of {}"
template = "a picture showing {}"
```

### Thay đổi batch size

Trong hàm `build_dataset()`:
```python
dataset, dataloader = build_dataset(project_root, preprocess, batch_size=8)  # Tăng batch size
```

### Thay đổi mô hình CLIP

Trong hàm `main()`:
```python
model, preprocess = clip.load("ViT-B/16", device=device)  # Sử dụng mô hình lớn hơn
# Các tùy chọn: "RN50", "RN101", "ViT-B/32", "ViT-B/16", "ViT-L/14"
```

### Thay đổi số lượng ảnh ví dụ

Trong hàm `main()`:
```python
show_examples(dataset, all_preds, all_labels, class_names, num_examples=10)  # Hiển thị 10 ảnh
```

## Đánh giá kết quả

- **Top-1 Accuracy:** Tỷ lệ phần trăm dự đoán đúng trên toàn bộ dataset
- **Confusion Matrix:** Ma trận cho biết mô hình nhầm lẫn giữa các lớp như thế nào
- **Per-class Accuracy:** Độ chính xác riêng cho từng lớp (hữu ích khi dataset không cân bằng)

## Lưu ý

- Lần chạy đầu tiên sẽ mất thời gian để tải mô hình CLIP (~350MB)
- Mô hình được cache tại `~/.cache/clip/`
- Nếu không có GPU, chương trình sẽ tự động chạy trên CPU
- Backend matplotlib được đặt là 'Agg' nên không cần GUI để chạy

## Troubleshooting

### Lỗi CUDA out of memory
Giảm batch size xuống:
```python
batch_size=2  # hoặc 1
```

### Lỗi không tìm thấy thư mục images
Đảm bảo:
- Thư mục `images/` nằm cùng cấp với file `zero_shot.py`
- Có 10 thư mục con tương ứng 10 classes CIFAR-10
- Mỗi thư mục có ít nhất 1 ảnh (.jpg, .jpeg, .png)

### Lỗi khi cài đặt CLIP
Thử cài đặt trực tiếp:
```bash
pip install git+https://github.com/openai/CLIP.git
```

## So sánh với các phương pháp khác

**Ưu điểm Zero-shot:**
- ✅ Không cần ảnh training
- ✅ Dễ dàng thêm classes mới
- ✅ Nhanh nhất (1 prompt/class)

**Nhược điểm:**
- ⚠️ Accuracy thấp hơn Few-shot
- ⚠️ Phụ thuộc vào prompt design

## Tài liệu tham khảo

- [CLIP Paper](https://arxiv.org/abs/2103.00020) - Learning Transferable Visual Models From Natural Language Supervision
- [CLIP GitHub](https://github.com/openai/CLIP) - Official implementation
- [OpenAI Blog](https://openai.com/blog/clip/) - CLIP: Connecting text and images
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html) - Dataset documentation

---

**Tác giả:** Gia Huy  
**GitHub:** [@HuyTranGia14](https://github.com/HuyTranGia14)  
**Repository:** [clip-image-classification](https://github.com/HuyTranGia14/clip-image-classification)