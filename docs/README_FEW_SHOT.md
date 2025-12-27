# Few-Shot Learning với CLIP

## 📖 Giới thiệu

Chương trình này triển khai **Few-Shot Learning** sử dụng CLIP model để phân loại ảnh với số lượng examples hạn chế. Thay vì sử dụng prompt templates như phương pháp Prompt Ensemble, Few-Shot Learning sử dụng một số ít ảnh mẫu (support set) để học đặc trưng của mỗi class.

### **Few-Shot Learning là gì?**

Few-Shot Learning là kỹ thuật học máy cho phép model phân loại các đối tượng mới với **rất ít dữ liệu training**:
- **1-shot**: Chỉ cần 1 ảnh mẫu cho mỗi class
- **5-shot**: Sử dụng 5 ảnh mẫu cho mỗi class  
- **10-shot**: Sử dụng 10 ảnh mẫu cho mỗi class

### **Cách hoạt động:**

1. **Xây dựng Support Set**: Chọn K ảnh mẫu cho mỗi class (K = 1, 5, hoặc 10)
2. **Encode Support Images**: Sử dụng CLIP để encode tất cả ảnh support thành feature vectors
3. **Tính Prototypes**: Mỗi class có 1 prototype = trung bình của K support features
4. **Classification**: So sánh query image với các prototypes, chọn class có độ tương đồng cao nhất

## 📂 Cấu trúc dữ liệu

```
clip_prompt_ensemble/
├── few_shot.py              # Chương trình Few-Shot Learning
├── requirements_fewshot.txt # Dependencies
├── README_FEWSHOT.md        # File hướng dẫn này
└── images/                  # Dataset
    ├── airplane/            # Ít nhất 10 ảnh
    ├── automobile/          # Ít nhất 10 ảnh
    ├── bird/
    ├── cat/
    ├── deer/
    ├── dog/
    ├── frog/
    ├── horse/
    ├── ship/
    └── truck/
```

**Lưu ý**: Mỗi class cần ít nhất 10 ảnh để hỗ trợ cả 3 chế độ (1-shot, 5-shot, 10-shot).

## 🚀 Cài đặt

### **Bước 1: Tạo virtual environment**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### **Bước 2: Cài đặt dependencies**

```powershell
pip install -r requirements_fewshot.txt
```

Hoặc cài thủ công:

```powershell
pip install torch>=1.12.0 torchvision>=0.13.0
pip install git+https://github.com/openai/CLIP.git
pip install ftfy>=6.1.0 regex>=2022.0.0 tqdm>=4.64.0
pip install Pillow>=9.0.0 matplotlib>=3.5.0 numpy>=1.21.0
```

### **Bước 3: Chuẩn bị dữ liệu**

Đặt ảnh test vào thư mục `images/` với cấu trúc:
- 10 folders tương ứng 10 classes
- Mỗi folder chứa ít nhất 10 ảnh (định dạng .jpg, .jpeg, .png)

## 📖 Hướng dẫn sử dụng

### **Khởi chạy chương trình:**

```powershell
python src/few_shot.py
```

### **Menu chính:**

Khi chương trình khởi động, bạn sẽ thấy menu:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    🎯 FEW-SHOT LEARNING WITH CLIP 🎯
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 MENU:
  [1] Chọn K-shot (hiện tại: ?)
  [2] Random test 1 ảnh
  [3] Test toàn bộ dataset
  [4] Xem statistics
  [q] Thoát

>>> 
```

### **Các chức năng:**

#### **[1] Chọn K-shot**

Chọn số lượng support examples cho mỗi class:
- **1**: 1-shot learning (1 ảnh/class)
- **2**: 5-shot learning (5 ảnh/class)
- **3**: 10-shot learning (10 ảnh/class)

Sau khi chọn, hệ thống sẽ:
- Build support set với K ảnh ngẫu nhiên cho mỗi class
- Encode tất cả support images
- Tính prototypes (mean features) cho mỗi class

#### **[2] Random test 1 ảnh**

- Hệ thống random chọn 1 class
- Random chọn 1 ảnh từ class đó (không nằm trong support set)
- Phân loại ảnh và hiển thị:
  - Predicted class và confidence
  - Top-5 predictions với xác suất
  - Kết quả: Đúng (✅) hoặc Sai (❌)
  - Biểu đồ visualization (lưu vào `fewshot_result.png`)

#### **[3] Test toàn bộ dataset**

- Test tất cả các ảnh trong dataset (trừ support set)
- Hiển thị:
  - Progress bar
  - Accuracy tổng thể
  - Confusion matrix (nếu có)
  - Thống kê chi tiết cho từng class

#### **[4] Xem statistics**

Hiển thị thống kê tích lũy:
- Tổng số test
- Số test đúng/sai
- Accuracy hiện tại
- Breakdown theo K-shot (nếu đã test nhiều K)

## 🎯 Quy trình sử dụng tiêu chuẩn

1. **Khởi chạy**: `python src/few_shot.py`
2. **Chọn K-shot**: Nhập `1` → Chọn `1`, `2`, hoặc `3`
3. **Test ngẫu nhiên**: Nhập `2` → Xem kết quả
4. **Lặp lại**: Test nhiều lần hoặc thử các K-shot khác nhau
5. **Xem thống kê**: Nhập `4` → Xem performance
6. **Thoát**: Nhập `q`

## 📊 Output

### **Console Output:**
- Thông tin chi tiết về support set
- Quá trình encoding và tính prototypes
- Kết quả phân loại với confidence scores
- Top-5 predictions
- Statistics tích lũy

### **File Output:**
- `fewshot_result.png`: Biểu đồ visualization gồm:
  - Query image
  - Bar chart xác suất 10 classes
  - Highlight class được dự đoán

## 🔬 So sánh với Prompt Ensemble

| Tiêu chí | Prompt Ensemble | Few-Shot Learning |
|----------|----------------|-------------------|
| **Input** | Text prompts | Support images |
| **Training data** | Không cần ảnh train | Cần K ảnh/class |
| **Flexibility** | Dễ thay đổi prompts | Cần chuẩn bị ảnh |
| **Performance** | Tốt với classes phổ biến | Tốt với classes hiếm |
| **Use case** | Zero-shot classification | Few-shot adaptation |

## ⚙️ Yêu cầu hệ thống

- **Python**: 3.8 trở lên
- **RAM**: Tối thiểu 4GB (8GB khuyến nghị)
- **GPU**: Không bắt buộc (nhưng nhanh hơn với CUDA)
- **Storage**: ~500MB cho CLIP model + dataset

## 🛠️ Troubleshooting

### **Lỗi: "Không đủ ảnh cho K-shot"**
- Đảm bảo mỗi class có ít nhất 10 ảnh
- Kiểm tra định dạng file (.jpg, .jpeg, .png)

### **Lỗi: "CLIP model download failed"**
- Kiểm tra kết nối internet
- Thử cài lại: `pip install --upgrade git+https://github.com/openai/CLIP.git`

### **Chương trình chạy chậm:**
- Cài đặt PyTorch với CUDA support
- Giảm số lượng K-shot
- Giảm số lượng ảnh test

## 📝 Ghi chú kỹ thuật

### **Prototype-based Classification:**

```
Prototype_c = (1/K) * Σ(CLIP_encode(support_image_i))
```

### **Similarity Scoring:**

```
Similarity = cosine_similarity(query_features, prototype)
Probability = softmax(similarity * temperature)
```

### **Temperature Scaling:**

Temperature = 100 (giống CLIP standard) để scale similarity scores trước khi softmax.

## 📚 Tài liệu tham khảo

- [CLIP Paper](https://arxiv.org/abs/2103.00020) - Learning Transferable Visual Models
- [Few-Shot Learning Survey](https://arxiv.org/abs/1904.05046)
- [Prototypical Networks](https://arxiv.org/abs/1703.05175)
