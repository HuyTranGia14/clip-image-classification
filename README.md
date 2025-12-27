# 🎯 Khảo sát Phân lớp Ảnh Sử dụng Mô hình Ngôn ngữ Trực quan CLIP

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![CLIP](https://img.shields.io/badge/CLIP-OpenAI-green.svg)](https://github.com/openai/CLIP)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Dự án nghiên cứu:** So sánh hiệu quả của ba phương pháp phân loại ảnh sử dụng CLIP: Zero-shot Learning, Few-shot Learning, và Prompt Ensemble

<div align="center">
  <img src="https://github.com/openai/CLIP/raw/main/CLIP.png" alt="CLIP Architecture" width="600"/>
</div>

## 📖 Giới thiệu

Dự án này triển khai và so sánh ba phương pháp phân loại ảnh tiên tiến sử dụng mô hình **CLIP (Contrastive Language-Image Pre-training)** của OpenAI:

1. **Zero-shot Learning** - Phân loại mà không cần dữ liệu huấn luyện
2. **Few-shot Learning** - Học từ số lượng mẫu hạn chế (1-shot, 5-shot, 10-shot)
3. **Prompt Ensemble** - Kết hợp nhiều prompt templates để tăng độ chính xác

### 🎓 Mục tiêu nghiên cứu

- So sánh hiệu quả của các phương pháp trên tập CIFAR-10
- Đánh giá khả năng zero-shot và few-shot của CLIP
- Khảo sát ảnh hưởng của prompt engineering
- Phân tích ensemble methods (Mean, Max, Weighted)

## ✨ Tính năng nổi bật

### 🔹 Zero-shot Learning
- ✅ Phân loại không cần training data
- ✅ Sử dụng text prompts đơn giản
- ✅ Confusion matrix và per-class accuracy
- ✅ Visualization kết quả

### 🔹 Few-shot Learning
- ✅ Hỗ trợ 1-shot, 5-shot, 10-shot
- ✅ Prototype-based classification
- ✅ Interactive menu system
- ✅ Real-time statistics tracking

### 🔹 Prompt Ensemble
- ✅ 10 prompt templates đa dạng
- ✅ 3 ensemble methods (Mean, Max, Weighted)
- ✅ So sánh trực tiếp các methods
- ✅ Similarity matrix visualization

## 📂 Cấu trúc dự án

```
clip-image-classification/
│
├── README.md                          # File này - Hướng dẫn tổng quan
│
├── 📁 Zero-shot Learning/
│   ├── zero_shot.py                   # Implementation
│   ├── README_ZERO_SHOT.md            # Hướng dẫn chi tiết
│   └── requirements_zero_shot.txt     # Dependencies
│
├── 📁 Few-shot Learning/
│   ├── few_shot.py                    # Implementation
│   ├── README_FEW_SHOT.md             # Hướng dẫn chi tiết
│   └── requirements_few_shot.txt      # Dependencies
│
├── 📁 Prompt Ensemble/
│   ├── prompt_ensemble.py             # Implementation
│   ├── README_PROMPT_ENSEMBLE.md      # Hướng dẫn chi tiết
│   └── requirements_prompt_ensemble.txt # Dependencies
│
└── 📁 Dataset/
    ├── images - zeroshot/             # Dataset cho zero-shot (4 classes)
    │   ├── airplane/
    │   ├── car/
    │   ├── cat/
    │   └── dog/
    │
    └── images - fs&pe/                # Dataset cho few-shot & prompt ensemble (10 classes)
        ├── airplane/
        ├── automobile/
        ├── bird/
        ├── cat/
        ├── deer/
        ├── dog/
        ├── frog/
        ├── horse/
        ├── ship/
        └── truck/
```

## 🚀 Bắt đầu nhanh

### 📋 Yêu cầu hệ thống

- **Python:** 3.7 trở lên
- **RAM:** Tối thiểu 4GB (khuyến nghị 8GB)
- **GPU:** Không bắt buộc (có CUDA sẽ nhanh hơn)
- **Storage:** ~500MB cho CLIP model + dataset

### ⚡ Cài đặt

#### 1️⃣ Clone repository

```bash
git clone https://github.com/HuyTranGia14/clip-image-classification.git
cd clip-image-classification
```

#### 2️⃣ Tạo virtual environment (khuyến nghị)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

#### 3️⃣ Cài đặt dependencies

**Chọn một trong ba phương pháp:**

```bash
# Zero-shot Learning
pip install -r requirements_zero_shot.txt

# Few-shot Learning
pip install -r requirements_few_shot.txt

# Prompt Ensemble
pip install -r requirements_prompt_ensemble.txt
```

**Hoặc cài đặt tất cả (để chạy cả 3 phương pháp):**

```bash
pip install torch torchvision numpy pillow scikit-learn matplotlib
pip install git+https://github.com/openai/CLIP.git
pip install ftfy regex tqdm
```

### 🎮 Chạy thử

#### **Zero-shot Learning**

```bash
python src/zero_shot.py
```

**Output:**
- Console: Accuracy, confusion matrix, per-class stats
- Files: `results/confusion_matrix.png`, `results/example_XX_*.png`

#### **Few-shot Learning**

```bash
python src/few_shot.py
```

**Interactive Menu:**
1. Chọn K-shot (1, 5, hoặc 10)
2. Random test hoặc continuous test
3. Xem statistics

#### **Prompt Ensemble**

```bash
python src/prompt_ensemble.py
```

**Interactive Menu:**
1. Random test với ensemble method
2. So sánh các ensemble methods
3. Xem statistics

## 📊 Dataset

### **Chuẩn bị dữ liệu**

Dự án sử dụng dataset từ CIFAR-10. Bạn cần chuẩn bị ảnh theo cấu trúc:

```
images/
├── airplane/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── car/ (hoặc automobile/)
│   └── ...
└── ...
```

**Lưu ý:**
- Zero-shot: Cần 4 classes (airplane, car, cat, dog)
- Few-shot & Prompt Ensemble: Cần 10 classes CIFAR-10
- Mỗi class nên có ít nhất 10-20 ảnh
- Định dạng hỗ trợ: `.jpg`, `.jpeg`, `.png`

### **Download dataset mẫu**

```bash
# Tải CIFAR-10 images (nếu bạn chưa có)
# Link: https://www.cs.toronto.edu/~kriz/cifar.html
```

## 🔬 Phương pháp nghiên cứu

### 1️⃣ Zero-shot Learning

**Nguyên lý:**
- Sử dụng text prompts đơn giản: "a photo of a {class}"
- CLIP encode cả text và image thành vector embeddings
- So sánh similarity giữa image và text features
- Chọn class có similarity cao nhất

**Ưu điểm:**
- Không cần training data
- Áp dụng nhanh cho classes mới
- Đơn giản, dễ triển khai

**Nhược điểm:**
- Phụ thuộc vào quality của prompts
- Accuracy thấp hơn với classes phức tạp

### 2️⃣ Few-shot Learning

**Nguyên lý:**
- Sử dụng K ảnh mẫu (support set) cho mỗi class
- Tính prototype = mean của K support features
- Classify query image bằng nearest prototype

**Ưu điểm:**
- Học nhanh từ ít examples
- Tốt với classes hiếm/không phổ biến
- Flexible với số lượng examples

**Nhược điểm:**
- Cần chuẩn bị support set
- Performance phụ thuộc vào quality của support examples

### 3️⃣ Prompt Ensemble

**Nguyên lý:**
- Sử dụng nhiều prompt templates khác nhau
- Ensemble aggregation: Mean, Max, Weighted
- Kết hợp kết quả từ tất cả prompts

**Ưu điểm:**
- Robust hơn single prompt
- Không cần support images
- Flexible với prompt design

**Nhược điểm:**
- Chậm hơn (nhiều prompts)
- Cần thiết kế prompts tốt

## 📈 Kết quả thực nghiệm

### **So sánh trên CIFAR-10 (4 classes)**

| Phương pháp | Accuracy | Thời gian | Pros | Cons |
|-------------|----------|-----------|------|------|
| **Zero-shot** | ~65-75% | Nhanh nhất | Đơn giản, không cần data | Accuracy thấp |
| **Few-shot (1-shot)** | ~70-80% | Trung bình | Học nhanh | Cần support set |
| **Few-shot (5-shot)** | ~75-85% | Trung bình | Cân bằng tốt | Cần nhiều ảnh hơn |
| **Few-shot (10-shot)** | ~80-90% | Trung bình | Accuracy cao nhất | Cần nhiều ảnh |
| **Prompt Ensemble (Mean)** | ~70-80% | Chậm | Robust | Chậm, design prompts |
| **Prompt Ensemble (Max)** | ~65-75% | Chậm | Nhạy với best prompt | Không ổn định |
| **Prompt Ensemble (Weighted)** | ~75-85% | Chậm | Flexible | Cần tune weights |

**Kết luận:**
- Few-shot (10-shot) cho accuracy cao nhất
- Prompt Ensemble (Mean) cân bằng tốt giữa performance và robustness
- Zero-shot phù hợp cho rapid prototyping

## 🛠️ Tùy chỉnh

### **Thay đổi CLIP model**

```python
# Trong mỗi file .py, thay đổi:
model, preprocess = clip.load("ViT-B/32", device=device)

# Các tùy chọn:
# "RN50", "RN101", "RN50x4", "RN50x16", "RN50x64"
# "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"
```

### **Thêm classes mới**

1. Thêm folder mới vào `images/`
2. Update `CLASS_NAMES` trong file Python:

```python
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
    "lion", "tiger"  # Thêm classes mới
]
```

### **Thay đổi prompt templates**

```python
# Trong prompt_ensemble.py
PROMPT_TEMPLATES = [
    "a photo of a {}",
    "an image of {}",
    "a picture showing {}",
    # Thêm templates của bạn...
]
```

## 📚 Tài liệu tham khảo

### **Papers**

1. **CLIP (2021)** - Learning Transferable Visual Models From Natural Language Supervision
   - [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
   - OpenAI Research

2. **Few-Shot Learning**
   - Prototypical Networks: [arXiv:1703.05175](https://arxiv.org/abs/1703.05175)
   - Matching Networks: [arXiv:1606.04080](https://arxiv.org/abs/1606.04080)

3. **Prompt Engineering**
   - CoOp: [arXiv:2109.01134](https://arxiv.org/abs/2109.01134)
   - CLIP-Adapter: [arXiv:2110.04544](https://arxiv.org/abs/2110.04544)

### **Resources**

- [CLIP GitHub](https://github.com/openai/CLIP) - Official implementation
- [OpenAI Blog](https://openai.com/blog/clip/) - CLIP announcement
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

## 🤝 Đóng góp

Contributions are welcome! Vui lòng tạo pull request hoặc issue nếu bạn:
- Tìm thấy bugs
- Có ý tưởng cải tiến
- Muốn thêm features mới
- Cải thiện documentation

### **Development Setup**

```bash
# Fork repository
git clone https://github.com/HuyTranGia14/clip-image-classification.git
cd clip-image-classification

# Tạo branch mới
git checkout -b feature/your-feature-name

# Commit changes
git commit -m "Add your feature"

# Push và tạo PR
git push origin feature/your-feature-name
```

## 📝 License

Dự án này được phân phối dưới giấy phép **MIT License**. Xem file [LICENSE](LICENSE) để biết thêm chi tiết.

```
MIT License

Copyright (c) 2025 Gia Huy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files...
```

## 📧 Liên hệ

**Tác giả:** Gia Huy
- 📧 Email: trangiahuy14092003@gmail.com
- 🐙 GitHub: [@HuyTranGia14](https://github.com/HuyTranGia14)

**Dự án:** [https://github.com/HuyTranGia14/clip-image-classification](https://github.com/HuyTranGia14/clip-image-classification)

---

## 🙏 Acknowledgments

- **OpenAI** - Phát triển CLIP model
- **PyTorch Team** - Deep learning framework
- **CIFAR-10** - Dataset benchmark
- Cộng đồng AI/ML Việt Nam

---

<div align="center">
  <p><strong>⭐ Nếu dự án hữu ích, đừng quên star repo này! ⭐</strong></p>
  <p>Made with ❤️ by Gia Huy</p>
</div>

## 📋 Changelog

### Version 1.0.0 (2025-01-XX)
- ✅ Initial release
- ✅ Zero-shot Learning implementation
- ✅ Few-shot Learning (1/5/10-shot)
- ✅ Prompt Ensemble (Mean/Max/Weighted)
- ✅ Interactive menu systems
- ✅ Comprehensive documentation

---

**Happy Coding! 🚀**
