"""
DEMO: Prompt Engineering & Ensemble Methods cho CLIP
Phân loại ảnh sử dụng nhiều prompt templates
"""

import torch
import clip
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Sử dụng backend không cần GUI
import matplotlib.pyplot as plt

# ==============================================================================
# BƯỚC 1: KHỞI TẠO - Load CLIP model
# ==============================================================================
print("=" * 70)
print("BƯỚC 1: Đang tải CLIP model...")
print("=" * 70)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Load model CLIP ViT-B/32
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()  # Chế độ evaluation
print("✓ Model loaded successfully!\n")


# ==============================================================================
# BƯỚC 2: THIẾT LẬP PROMPTS VÀ CLASSES
# ==============================================================================
print("=" * 70)
print("BƯỚC 2: Chuẩn bị Prompt Templates và Classes")
print("=" * 70)

# Định nghĩa classes cần phân loại (ví dụ: CIFAR-10)
class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Định nghĩa prompt templates (bắt đầu với 10 templates đơn giản)
templates = [
    "a photo of a {}",
    "a blurry photo of a {}",
    "a bright photo of a {}",
    "a dark photo of a {}",
    "a close-up photo of a {}",
    "a photo of many {}",
    "a photo of the large {}",
    "a photo of the small {}",
    "a black and white photo of a {}",
    "a cropped photo of a {}",
]

M = len(templates)  # Số templates
N = len(class_names)  # Số classes

print(f"Số templates: {M}")
print(f"Số classes: {N}")
print(f"Tổng số prompts: {M * N} = {M} × {N}\n")

print("Templates:")
for i, template in enumerate(templates[:5]):  # Hiển thị 5 template đầu
    print(f"  {i+1}. {template}")
print(f"  ... và {M-5} templates khác\n")


# ==============================================================================
# BƯỚC 3: ENCODE TẤT CẢ TEXT PROMPTS (làm 1 lần duy nhất)
# ==============================================================================
print("=" * 70)
print("BƯỚC 3: Encoding tất cả Text Prompts...")
print("=" * 70)

text_features = []

for i, template in enumerate(templates):
    print(f"Đang xử lý template {i+1}/{M}: '{template[:30]}...'")
    
    for class_name in class_names:
        # Tạo prompt hoàn chỉnh
        text = template.format(class_name)
        
        # Tokenize
        tokens = clip.tokenize([text]).to(device)
        
        # Encode
        with torch.no_grad():
            text_feat = model.encode_text(tokens)
            # Normalize
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        
        text_features.append(text_feat)

# Chuyển thành tensor và reshape
text_features = torch.cat(text_features, dim=0)  # [M*N, 512]
text_features = text_features.view(M, N, -1)  # [M, N, 512]

print(f"\n✓ Text features shape: {text_features.shape}")
print(f"  - {M} templates")
print(f"  - {N} classes")
print(f"  - 512 dimensions\n")


# ==============================================================================
# HÀM PHỤ TRỢ: PHÂN LOẠI ẢNH VỚI ENSEMBLE
# ==============================================================================

def classify_image(image_path, ensemble_method="mean", show_details=True):
    """
    Phân loại ảnh sử dụng Prompt Ensemble
    
    Args:
        image_path: Đường dẫn đến ảnh
        ensemble_method: "mean" | "max" | "weighted"
        show_details: Hiển thị chi tiết hay không
    
    Returns:
        predicted_class, confidence, all_probs
    """
    
    if show_details:
        print("=" * 70)
        print(f"PHÂN LOẠI ẢNH: {image_path}")
        print("=" * 70)
    
    # ─────── STEP 1: Load và preprocess ảnh ───────
    image = Image.open(image_path).convert('RGB')
    image_input = preprocess(image).unsqueeze(0).to(device)
    
    if show_details:
        print(f"✓ Image loaded: {image.size}")
        print(f"✓ Preprocessed: {image_input.shape}\n")
    
    # ─────── STEP 2: Encode ảnh ───────
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        # Normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    if show_details:
        print(f"✓ Image features: {image_features.shape}")
        print(f"✓ Normalized: ||v|| = {image_features.norm().item():.4f}\n")
    
    # ─────── STEP 3: Tính Similarity Matrix ───────
    # image_features: [1, 512]
    # text_features: [M, N, 512]
    
    similarity = torch.zeros(M, N).to(device)
    
    for i in range(M):  # Mỗi template
        for j in range(N):  # Mỗi class
            # Dot product
            sim = (image_features @ text_features[i, j]).item()
            # Scale by 100 (temperature)
            similarity[i, j] = 100.0 * sim
    
    if show_details:
        print(f"✓ Similarity matrix computed: {similarity.shape}")
        print(f"\nSimilarity Matrix (Top-3 classes for first 3 templates):")
        
        # Hiển thị top-3 cho mỗi template
        for i in range(min(3, M)):
            top3_indices = similarity[i].topk(3).indices.cpu().numpy()
            top3_values = similarity[i].topk(3).values.cpu().numpy()
            print(f"  Template {i+1}: ", end="")
            for idx, val in zip(top3_indices, top3_values):
                print(f"{class_names[idx]}={val:.1f}  ", end="")
            print()
        print()
    
    # ─────── STEP 4: Ensemble ───────
    if ensemble_method == "mean":
        # Trung bình cộng
        ensemble_scores = similarity.mean(dim=0)  # [N]
        if show_details:
            print("✓ Ensemble method: MEAN (Trung bình cộng)")
    
    elif ensemble_method == "max":
        # Lấy max
        ensemble_scores = similarity.max(dim=0)[0]  # [N]
        if show_details:
            print("✓ Ensemble method: MAX (Lấy giá trị lớn nhất)")
    
    elif ensemble_method == "weighted":
        # Weighted sum (template cuối có trọng số cao hơn)
        weights = torch.linspace(0.5, 1.5, M).to(device)
        weights = weights / weights.sum()  # Normalize weights
        ensemble_scores = (similarity * weights.unsqueeze(1)).sum(dim=0)
        if show_details:
            print("✓ Ensemble method: WEIGHTED")
            print(f"  Weights: {weights.cpu().numpy()}")
    
    else:
        raise ValueError(f"Unknown ensemble method: {ensemble_method}")
    
    if show_details:
        print(f"\n✓ Ensemble scores shape: {ensemble_scores.shape}")
    
    # ─────── STEP 5: Softmax → Probabilities ───────
    probabilities = torch.softmax(ensemble_scores, dim=0)
    
    # ─────── STEP 6: Prediction ───────
    predicted_idx = probabilities.argmax().item()
    confidence = probabilities[predicted_idx].item()
    predicted_class = class_names[predicted_idx]
    
    if show_details:
        print("\n" + "=" * 70)
        print("KẾT QUẢ PHÂN LOẠI")
        print("=" * 70)
        print(f"Predicted Class: {predicted_class.upper()}")
        print(f"Confidence: {confidence * 100:.2f}%")
        print(f"\nTop-5 Predictions:")
        
        # Top-5
        top5_probs, top5_indices = probabilities.topk(5)
        for i, (prob, idx) in enumerate(zip(top5_probs, top5_indices)):
            bar = "█" * int(prob.item() * 50)
            print(f"  {i+1}. {class_names[idx]:12s}: {prob.item()*100:6.2f}% {bar}")
        print()
    
    return predicted_class, confidence, probabilities


# ==============================================================================
# HÀM VISUALIZE KẾT QUẢ
# ==============================================================================

def visualize_results(image_path, probabilities):
    """
    Vẽ biểu đồ kết quả phân loại
    """
    # Load ảnh
    image = Image.open(image_path)
    
    # Tạo figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Ảnh gốc
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title('Input Image', fontsize=14, fontweight='bold')
    
    # Plot 2: Bar chart probabilities
    probs_np = probabilities.cpu().numpy()
    colors = ['green' if p == probs_np.max() else 'skyblue' for p in probs_np]
    
    ax2.barh(class_names, probs_np * 100, color=colors)
    ax2.set_xlabel('Probability (%)', fontsize=12)
    ax2.set_title('Classification Probabilities', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # Highlight predicted class
    max_idx = probs_np.argmax()
    ax2.text(probs_np[max_idx] * 100 + 2, max_idx, 
             f'{probs_np[max_idx]*100:.1f}%', 
             va='center', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    
    # Lưu file vào folder results/
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    output_path = os.path.join(results_dir, 'classification_result.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Đã lưu biểu đồ vào '{output_path}'")
    plt.close()  # Đóng figure để giải phóng bộ nhớ
    # plt.show()  # Tắt GUI display


# ==============================================================================
# SO SÁNH CÁC ENSEMBLE METHODS
# ==============================================================================

def compare_ensemble_methods(image_path):
    """
    So sánh kết quả của 3 ensemble methods
    """
    print("\n" + "=" * 70)
    print("SO SÁNH CÁC ENSEMBLE METHODS")
    print("=" * 70)
    
    methods = ["mean", "max", "weighted"]
    results = {}
    
    for method in methods:
        print(f"\n{'─'*70}")
        print(f"Method: {method.upper()}")
        print(f"{'─'*70}")
        
        pred_class, conf, probs = classify_image(
            image_path, 
            ensemble_method=method, 
            show_details=False
        )
        
        results[method] = {
            'class': pred_class,
            'confidence': conf,
            'probabilities': probs
        }
        
        print(f"Prediction: {pred_class} ({conf*100:.2f}%)")
    
    # Tóm tắt
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Method':<12} {'Predicted Class':<15} {'Confidence':<12}")
    print("-" * 70)
    for method in methods:
        print(f"{method:<12} {results[method]['class']:<15} "
              f"{results[method]['confidence']*100:>6.2f}%")
    print()
    
    return results


# ==============================================================================
# MAIN: DEMO
# ==============================================================================

if __name__ == "__main__":
    import os
    import random
    from pathlib import Path
    
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  DEMO: PROMPT ENGINEERING & ENSEMBLE FOR IMAGE CLASSIFICATION  ".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70 + "\n")
    
    # Khởi tạo biến đếm
    total_tests = 0
    correct_predictions = 0
    
    # ───────────────────────────────────────────────────────────────
    # VÒNG LẶP CHÍNH: Test liên tục cho đến khi thoát
    # ───────────────────────────────────────────────────────────────
    
    while True:
        print("\n" + "=" * 70)
        print("CHỌN CLASS ĐỂ TEST (Hệ thống sẽ random chọn 1 ảnh)")
        print("=" * 70)
        
        # Kiểm tra số lượng ảnh có sẵn cho mỗi class
        # Lấy đường dẫn gốc của project (parent của src/)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        images_base = os.path.join(project_root, 'images')
        class_image_counts = {}
        
        for i, class_name in enumerate(class_names, 1):
            # Tìm tất cả ảnh của class này
            class_folder = os.path.join(images_base, class_name)
            if os.path.exists(class_folder):
                image_files = list(Path(class_folder).glob('*.jpg')) + \
                             list(Path(class_folder).glob('*.jpeg')) + \
                             list(Path(class_folder).glob('*.png'))
                class_image_counts[class_name] = len(image_files)
            else:
                class_image_counts[class_name] = 0
        
        print("\n10 classes có sẵn:")
        print("-" * 70)
        
        for i, class_name in enumerate(class_names, 1):
            count = class_image_counts[class_name]
            status = f"✓ ({count} ảnh)" if count > 0 else "✗ (chưa có ảnh)"
            print(f"  {i:2d}. {class_name:12s} - {status}")
        
        print("-" * 70)
        
        # Hiển thị thống kê nếu đã có test
        if total_tests > 0:
            accuracy = (correct_predictions / total_tests) * 100
            print(f"\n📊 Thống kê: {correct_predictions}/{total_tests} đúng ({accuracy:.1f}% accuracy)")
        
        print("\nNhập số (1-10) để chọn class, hoặc 'q' để thoát:")
        
        choice = input(">>> ").strip().lower()
        
        # Kiểm tra thoát
        if choice == 'q' or choice == 'quit' or choice == 'exit':
            break
        
        # Xác định class sẽ test
        if choice.isdigit() and 1 <= int(choice) <= 10:
            selected_class = class_names[int(choice) - 1]
            print(f"\n✓ Đã chọn class: {selected_class}")
        elif choice == '':
            # Random chọn class có ảnh
            available_classes = [c for c in class_names if class_image_counts[c] > 0]
            if not available_classes:
                print("\n✗ KHÔNG CÓ ẢNH NÀO!")
                print("Vui lòng chuẩn bị ảnh trong folder images/")
                continue
            selected_class = random.choice(available_classes)
            print(f"\n✓ Random chọn class: {selected_class}")
        else:
            print("\n✗ Lựa chọn không hợp lệ! Vui lòng nhập số 1-10 hoặc 'q' để thoát.")
            continue
        
        # Random chọn 1 ảnh từ class đã chọn
        class_folder = os.path.join(images_base, selected_class)
        
        if not os.path.exists(class_folder) or class_image_counts[selected_class] == 0:
            print(f"\n⚠ CẢNH BÁO: Không có ảnh cho class '{selected_class}'")
            print(f"Vui lòng đặt ảnh vào folder: {class_folder}")
            continue
        
        # Lấy danh sách ảnh
        image_files = list(Path(class_folder).glob('*.jpg')) + \
                     list(Path(class_folder).glob('*.jpeg')) + \
                     list(Path(class_folder).glob('*.png'))
        
        # Random chọn 1 ảnh
        selected_image = random.choice(image_files)
        image_path = str(selected_image)
        
        print(f"✓ Random chọn ảnh: {selected_image.name}")
        print(f"  Có {len(image_files)} ảnh khả dụng cho class này")
        print(f"\nExpected class: {selected_class}")
        
        # ───────────────────────────────────────────────────────────────
        # TEST: Phân loại ảnh đã chọn
        # ───────────────────────────────────────────────────────────────
        
        # Phân loại với Mean Ensemble
        predicted_class, confidence, probabilities = classify_image(
            image_path, 
            ensemble_method="mean",
            show_details=True
        )
        
        # Cập nhật thống kê
        total_tests += 1
        is_correct = (predicted_class == selected_class)
        if is_correct:
            correct_predictions += 1
        
        # Kiểm tra kết quả
        print("\n" + "=" * 70)
        if is_correct:
            print("✅ DỰ ĐOÁN CHÍNH XÁC!")
        else:
            print(f"❌ DỰ ĐOÁN SAI!")
            print(f"   Expected: {selected_class}")
            print(f"   Predicted: {predicted_class}")
        print("=" * 70)
        
        # Visualize
        visualize_results(image_path, probabilities)
        
        # ───────────────────────────────────────────────────────────────
        # OPTION: So sánh các Ensemble Methods
        # ───────────────────────────────────────────────────────────────
        
        print("\n" + "=" * 70)
        print("So sánh các Ensemble Methods? (y/n)")
        compare_choice = input(">>> ").strip().lower()
        
        if compare_choice == 'y' or compare_choice == 'yes':
            compare_ensemble_methods(image_path)
        
        print("\n" + "=" * 70)
        print("📝 KẾT QUẢ:")
        print(f"  - Class được chọn: {selected_class}")
        print(f"  - Ảnh được test: {selected_image.name}")
        print(f"  - Predicted: {predicted_class} ({confidence*100:.1f}%)")
        print(f"  - Kết quả: {'✓ Đúng' if is_correct else '✗ Sai'}")
        print(f"  - File visualization: classification_result.png")
        print("=" * 70)
        
        # Hỏi tiếp tục hoặc thoát
        print("\nNhấn Enter để tiếp tục test class khác, hoặc 'q' để thoát:")
        continue_choice = input(">>> ").strip().lower()
        
        if continue_choice == 'q' or continue_choice == 'quit' or continue_choice == 'exit':
            break
    
    # ───────────────────────────────────────────────────────────────
    # KẾT THÚC: Hiển thị thống kê tổng kết
    # ───────────────────────────────────────────────────────────────
    
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "DEMO HOÀN THÀNH!".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    
    if total_tests > 0:
        accuracy = (correct_predictions / total_tests) * 100
        print("\n📊 THỐNG KÊ TỔNG KẾT:")
        print("=" * 70)
        print(f"  - Tổng số test: {total_tests}")
        print(f"  - Dự đoán đúng: {correct_predictions}")
        print(f"  - Dự đoán sai: {total_tests - correct_predictions}")
        print(f"  - Accuracy: {accuracy:.2f}%")
        print("=" * 70)
    
    print("\n✓ Cảm ơn đã sử dụng demo!")
    input("\nNhấn Enter để thoát...")