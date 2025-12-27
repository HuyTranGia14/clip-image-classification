"""
FEW-SHOT LEARNING với CLIP
================================
Phân loại ảnh sử dụng K support examples cho mỗi class (K=1,5,10)
Prototype-based classification: Tính mean của support features làm prototype

"""

import os
import random
import torch
import clip
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Backend không cần GUI
import matplotlib.pyplot as plt
import numpy as np


# ================================================================================
# CẤU HÌNH HỆ THỐNG
# ================================================================================

# 10 classes từ CIFAR-10
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Các tùy chọn K-shot
K_SHOT_OPTIONS = {
    '1': 1,   # 1-shot: 1 example/class
    '2': 5,   # 5-shot: 5 examples/class
    '3': 10   # 10-shot: 10 examples/class
}


# ================================================================================
# 1. LOAD CLIP MODEL
# ================================================================================

def load_clip_model():
    """Load CLIP ViT-B/32 model"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*80}")
    print(f"🔧 ĐANG KHỞI TẠO HỆ THỐNG FEW-SHOT LEARNING")
    print(f"{'='*80}")
    print(f"📌 Device: {device.upper()}")
    
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    
    print(f"✅ Đã load CLIP ViT-B/32 successfully!")
    print(f"📊 Embedding dimension: 512")
    print(f"🎯 Classes: {len(CLASS_NAMES)} (CIFAR-10)")
    
    return model, preprocess, device


# ================================================================================
# 2. BUILD SUPPORT SET
# ================================================================================

def build_support_set(images_dir, k_shot):
    """
    Tạo support set với K examples cho mỗi class
    
    Args:
        images_dir: Thư mục chứa ảnh (có 10 folders con)
        k_shot: Số lượng examples/class (1, 5, hoặc 10)
    
    Returns:
        support_set: Dict {class_name: [list of K image paths]}
    """
    support_set = {}
    
    print(f"\n{'='*80}")
    print(f"📁 ĐANG XÂY DỰNG SUPPORT SET ({k_shot}-SHOT)")
    print(f"{'='*80}")
    
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(images_dir, class_name)
        
        if not os.path.exists(class_dir):
            print(f"❌ Không tìm thấy folder: {class_name}")
            continue
        
        # Lấy tất cả ảnh trong folder
        all_images = [f for f in os.listdir(class_dir) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if len(all_images) < k_shot:
            print(f"⚠️  {class_name}: Chỉ có {len(all_images)} ảnh (cần {k_shot})")
            selected = all_images
        else:
            # Random chọn K ảnh
            selected = random.sample(all_images, k_shot)
        
        # Lưu đường dẫn đầy đủ
        support_set[class_name] = [
            os.path.join(class_dir, img) for img in selected
        ]
        
        print(f"✅ {class_name:12s}: {len(support_set[class_name])} examples")
    
    total_examples = sum(len(imgs) for imgs in support_set.values())
    print(f"\n📊 Tổng số examples: {total_examples} ảnh")
    print(f"📈 Expected: {len(CLASS_NAMES)} classes × {k_shot} = {len(CLASS_NAMES) * k_shot} ảnh")
    
    return support_set


# ================================================================================
# 3. ENCODE SUPPORT SET
# ================================================================================

def encode_support_set(model, preprocess, support_set, device):
    """
    Encode tất cả support images và tính prototypes
    
    Args:
        model: CLIP model
        preprocess: CLIP preprocessing function
        support_set: Dict {class_name: [image_paths]}
        device: 'cuda' or 'cpu'
    
    Returns:
        prototypes: Tensor [num_classes, 512] - Mean features cho mỗi class
        class_order: List of class names (để map index -> class)
    """
    print(f"\n{'='*80}")
    print(f"🔄 ĐANG ENCODE SUPPORT SET & TÍNH PROTOTYPES")
    print(f"{'='*80}")
    
    prototypes = []
    class_order = []
    
    with torch.no_grad():
        for class_name in CLASS_NAMES:
            if class_name not in support_set:
                continue
            
            image_paths = support_set[class_name]
            features_list = []
            
            # Encode từng ảnh trong support set
            for img_path in image_paths:
                image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
                features = model.encode_image(image)
                features = features / features.norm(dim=-1, keepdim=True)
                features_list.append(features)
            
            # Tính prototype = mean của all support features
            class_features = torch.cat(features_list, dim=0)
            prototype = class_features.mean(dim=0, keepdim=True)
            prototype = prototype / prototype.norm(dim=-1, keepdim=True)
            
            prototypes.append(prototype)
            class_order.append(class_name)
            
            print(f"✅ {class_name:12s}: {len(features_list)} examples → 1 prototype [512-dim]")
    
    prototypes = torch.cat(prototypes, dim=0)  # [num_classes, 512]
    
    print(f"\n📊 Prototypes shape: {list(prototypes.shape)}")
    print(f"✅ Đã tính {len(class_order)} prototypes successfully!")
    
    return prototypes, class_order


# ================================================================================
# 4. FEW-SHOT CLASSIFICATION
# ================================================================================

def few_shot_classify(model, preprocess, query_image_path, prototypes, class_order, device):
    """
    Phân loại 1 query image bằng Few-shot learning
    
    Args:
        model: CLIP model
        preprocess: CLIP preprocessing function
        query_image_path: Đường dẫn đến query image
        prototypes: Tensor [num_classes, 512]
        class_order: List of class names
        device: 'cuda' or 'cpu'
    
    Returns:
        predicted_class: Tên class được dự đoán
        confidence: Độ tin cậy (%)
        probabilities: Dict {class_name: probability}
    """
    # Load và encode query image
    query_image = preprocess(Image.open(query_image_path)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        query_features = model.encode_image(query_image)
        query_features = query_features / query_features.norm(dim=-1, keepdim=True)
        
        # Tính cosine similarity với các prototypes
        similarities = (query_features @ prototypes.T).squeeze(0)  # [num_classes]
        
        # Softmax để ra probabilities
        probabilities_tensor = torch.softmax(similarities * 100, dim=0)
        
        # Predicted class
        pred_idx = similarities.argmax().item()
        predicted_class = class_order[pred_idx]
        confidence = probabilities_tensor[pred_idx].item() * 100
        
        # Convert to dict
        probabilities = {
            class_order[i]: probabilities_tensor[i].item() * 100 
            for i in range(len(class_order))
        }
    
    return predicted_class, confidence, probabilities


# ================================================================================
# 5. VISUALIZATION
# ================================================================================

def visualize_result(query_image_path, predicted_class, probabilities, true_class=None):
    """
    Tạo visualization cho kết quả phân loại
    
    Args:
        query_image_path: Đường dẫn query image
        predicted_class: Class được dự đoán
        probabilities: Dict {class_name: probability}
        true_class: True label (nếu có)
    """
    # Tạo figure với 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Subplot 1: Hiển thị ảnh
    img = Image.open(query_image_path)
    ax1.imshow(img)
    ax1.axis('off')
    
    if true_class:
        status = "✅ CORRECT" if predicted_class == true_class else "❌ WRONG"
        title = f"True: {true_class} | Pred: {predicted_class}\n{status}"
    else:
        title = f"Predicted: {predicted_class}"
    
    ax1.set_title(title, fontsize=12, fontweight='bold')
    
    # Subplot 2: Bar chart xác suất
    classes = list(probabilities.keys())
    probs = list(probabilities.values())
    
    colors = ['#2ecc71' if c == predicted_class else '#3498db' for c in classes]
    
    bars = ax2.barh(classes, probs, color=colors)
    ax2.set_xlabel('Probability (%)', fontsize=10)
    ax2.set_title('Class Probabilities', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 100)
    
    # Thêm giá trị vào bars
    for bar, prob in zip(bars, probs):
        width = bar.get_width()
        ax2.text(width + 1, bar.get_y() + bar.get_height()/2, 
                f'{prob:.2f}%', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    
    # Lưu file vào folder results/
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    output_file = os.path.join(results_dir, 'few_shot_result.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_file


# ================================================================================
# 6. RANDOM TEST
# ================================================================================

def random_test(images_dir, model, preprocess, prototypes, class_order, device):
    """Random chọn 1 ảnh để test"""
    # Random chọn class
    true_class = random.choice(CLASS_NAMES)
    class_dir = os.path.join(images_dir, true_class)
    
    if not os.path.exists(class_dir):
        print(f"❌ Không tìm thấy folder: {true_class}")
        return None
    
    # Lấy list ảnh
    images = [f for f in os.listdir(class_dir) 
             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not images:
        print(f"❌ Không có ảnh trong folder: {true_class}")
        return None
    
    # Random chọn 1 ảnh
    selected_image = random.choice(images)
    query_path = os.path.join(class_dir, selected_image)
    
    print(f"\n{'='*80}")
    print(f"🎲 RANDOM TEST")
    print(f"{'='*80}")
    print(f"📁 True class: {true_class}")
    print(f"🖼️  Image: {selected_image}")
    print(f"📍 Path: {query_path}")
    
    # Classify
    print(f"\n🔄 Đang phân loại...")
    predicted_class, confidence, probabilities = few_shot_classify(
        model, preprocess, query_path, prototypes, class_order, device
    )
    
    # Kết quả
    print(f"\n{'='*80}")
    print(f"📊 KẾT QUẢ PHÂN LOẠI")
    print(f"{'='*80}")
    print(f"🎯 Predicted: {predicted_class}")
    print(f"📈 Confidence: {confidence:.2f}%")
    print(f"✅ True class: {true_class}")
    
    if predicted_class == true_class:
        print(f"🎉 Status: CORRECT ✅")
        result = True
    else:
        print(f"❌ Status: WRONG")
        result = False
    
    # Top-5
    print(f"\n📊 Top-5 Predictions:")
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    for i, (cls, prob) in enumerate(sorted_probs[:5], 1):
        marker = "✓" if cls == true_class else " "
        print(f"  {i}. {cls:12s} {prob:6.2f}% {marker}")
    
    # Visualization
    output_file = visualize_result(query_path, predicted_class, probabilities, true_class)
    print(f"\n💾 Đã lưu visualization: {output_file}")
    
    return result


# ================================================================================
# 7. STATISTICS TRACKING
# ================================================================================

class Statistics:
    def __init__(self):
        self.total_tests = 0
        self.correct = 0
        self.wrong = 0
    
    def update(self, is_correct):
        self.total_tests += 1
        if is_correct:
            self.correct += 1
        else:
            self.wrong += 1
    
    def get_accuracy(self):
        if self.total_tests == 0:
            return 0.0
        return (self.correct / self.total_tests) * 100
    
    def display(self):
        print(f"\n{'='*80}")
        print(f"📊 STATISTICS (Session)")
        print(f"{'='*80}")
        print(f"🔢 Total tests: {self.total_tests}")
        print(f"✅ Correct: {self.correct}")
        print(f"❌ Wrong: {self.wrong}")
        print(f"📈 Accuracy: {self.get_accuracy():.2f}%")
        print(f"{'='*80}")


# ================================================================================
# 8. MENU HỆ THỐNG
# ================================================================================

def display_menu():
    """Hiển thị menu chính"""
    print(f"\n{'='*80}")
    print(f"🎯 FEW-SHOT LEARNING - MENU CHÍNH")
    print(f"{'='*80}")
    print(f"1. Random test (1 ảnh)")
    print(f"2. Continuous test (nhiều ảnh liên tiếp)")
    print(f"3. Xem statistics")
    print(f"4. Đổi K-shot (hiện tại: {current_k_shot})")
    print(f"5. Rebuild support set")
    print(f"0. Thoát")
    print(f"{'='*80}")


def select_k_shot():
    """Chọn K-shot (1, 5, hoặc 10)"""
    print(f"\n{'='*80}")
    print(f"🎯 CHỌN K-SHOT")
    print(f"{'='*80}")
    print(f"1. 1-shot (1 example/class)")
    print(f"2. 5-shot (5 examples/class)")
    print(f"3. 10-shot (10 examples/class)")
    print(f"{'='*80}")
    
    while True:
        choice = input("👉 Nhập lựa chọn (1-3): ").strip()
        if choice in K_SHOT_OPTIONS:
            return K_SHOT_OPTIONS[choice]
        else:
            print("❌ Lựa chọn không hợp lệ. Vui lòng nhập 1, 2, hoặc 3.")


# ================================================================================
# 9. MAIN FUNCTION
# ================================================================================

def main():
    global current_k_shot
    
    # Setup - Lấy đường dẫn thư mục gốc của project (parent của src/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    images_dir = os.path.join(project_root, "images")
    
    # Load CLIP
    model, preprocess, device = load_clip_model()
    
    # Chọn K-shot
    current_k_shot = select_k_shot()
    
    # Build support set
    support_set = build_support_set(images_dir, current_k_shot)
    prototypes, class_order = encode_support_set(model, preprocess, support_set, device)
    
    # Statistics
    stats = Statistics()
    
    # Main loop
    while True:
        display_menu()
        choice = input("👉 Nhập lựa chọn: ").strip()
        
        if choice == '1':
            # Random test 1 ảnh
            result = random_test(images_dir, model, preprocess, prototypes, class_order, device)
            if result is not None:
                stats.update(result)
        
        elif choice == '2':
            # Continuous test
            print("\n🔄 CONTINUOUS TEST MODE")
            print("Nhấn Ctrl+C để dừng\n")
            
            try:
                count = 0
                while True:
                    count += 1
                    print(f"\n{'#'*80}")
                    print(f"TEST #{count}")
                    print(f"{'#'*80}")
                    
                    result = random_test(images_dir, model, preprocess, prototypes, class_order, device)
                    if result is not None:
                        stats.update(result)
                    
                    input("\n⏸️  Nhấn Enter để tiếp tục...")
            
            except KeyboardInterrupt:
                print("\n\n⏹️  Đã dừng continuous test.")
        
        elif choice == '3':
            # Xem statistics
            stats.display()
        
        elif choice == '4':
            # Đổi K-shot
            new_k_shot = select_k_shot()
            if new_k_shot != current_k_shot:
                current_k_shot = new_k_shot
                print(f"\n🔄 Đang rebuild support set với {current_k_shot}-shot...")
                support_set = build_support_set(images_dir, current_k_shot)
                prototypes, class_order = encode_support_set(model, preprocess, support_set, device)
                print(f"✅ Đã rebuild support set!")
        
        elif choice == '5':
            # Rebuild support set (với same K)
            print(f"\n🔄 Đang rebuild support set với {current_k_shot}-shot...")
            support_set = build_support_set(images_dir, current_k_shot)
            prototypes, class_order = encode_support_set(model, preprocess, support_set, device)
            print(f"✅ Đã rebuild support set!")
        
        elif choice == '0':
            # Thoát
            print(f"\n{'='*80}")
            print(f"👋 CẢM ƠN BẠN ĐÃ SỬ DỤNG FEW-SHOT LEARNING!")
            stats.display()
            print(f"{'='*80}\n")
            break
        
        else:
            print("❌ Lựa chọn không hợp lệ. Vui lòng thử lại.")


if __name__ == "__main__":
    current_k_shot = 5  # Default
    main()