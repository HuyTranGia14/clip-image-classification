import os
import torch
import clip
import numpy as np

from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Sử dụng backend không cần GUI
import matplotlib.pyplot as plt


# =========================
# 1. Tạo dataset & dataloader
# =========================
def build_dataset(project_root, preprocess, batch_size=4):
    # Sử dụng chung dataset 50 ảnh với few-shot và prompt ensemble
    # Dataset CIFAR-10 (10 classes) nằm trong thư mục "images"
    data_test_dir = os.path.join(project_root, "images")

    dataset = datasets.ImageFolder(data_test_dir, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return dataset, dataloader



# =========================
# 2. Tạo text_features từ prompt
# =========================
def build_text_features(class_names, model, device, template="a photo of a {}"):
    prompts = [template.format(c.replace("_", " ")) for c in class_names]
    print("prompts:", prompts)

    with torch.no_grad():
        text_tokens = clip.tokenize(prompts).to(device)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features


# =========================
# 3. Hàm evaluate zero-shot (CỐT LÕI)
# =========================
def zero_shot_evaluate(model, dataloader, text_features, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)

            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(
                dim=-1, keepdim=True
            )

            logits = image_features @ text_features.T
            preds = logits.argmax(dim=-1).cpu()
            labels = labels.cpu()

            all_preds.append(preds)
            all_labels.append(labels)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    acc = (all_preds == all_labels).float().mean().item()
    return acc, all_preds, all_labels


# =========================
# 4. [OPTIONAL] Confusion matrix + ví dụ minh hoạ
#     → Nếu muốn tối giản hoàn toàn, bạn có thể XÓA nguyên block này.
# =========================
def show_confusion_and_per_class_acc(all_labels, all_preds, class_names):
    y_true = all_labels.cpu().numpy()
    y_pred = all_preds.cpu().numpy()

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:")
    print(cm)

    cm_diag = np.diag(cm)
    per_class_counts = cm.sum(axis=1)
    per_class_acc = cm_diag / per_class_counts

    print("\nĐộ chính xác theo từng lớp:")
    for i, cls in enumerate(class_names):
        print(f"  - {cls}: {per_class_acc[i]:.4f}")

    fig, ax = plt.subplots()
    im = ax.imshow(cm)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix (Zero-shot CLIP)")
    fig.tight_layout()
    
    # Lưu confusion matrix vào folder results/
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    output_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nĐã lưu confusion matrix vào: {output_path}")
    plt.close()


def show_examples(dataset, all_preds, all_labels, class_names, num_examples=6):
    preds_list = all_preds.cpu().tolist()
    
    # Tạo thư mục results nếu chưa có
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("\nMột số ví dụ dự đoán:")
    for i in range(min(num_examples, len(dataset))):
        img_path, true_label_idx = dataset.samples[i]
        pred_label_idx = preds_list[i]

        true_name = class_names[true_label_idx]
        pred_name = class_names[pred_label_idx]

        print(
            f"{i:02d}. {os.path.basename(img_path)} | "
            f"True: {true_name}, Pred: {pred_name}"
        )

        img = Image.open(img_path).convert("RGB")
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"True: {true_name} | Pred: {pred_name}")
        
        # Lưu ảnh ví dụ vào folder results/
        example_output = os.path.join(results_dir, f"example_{i:02d}_{true_name}_pred_{pred_name}.png")
        plt.savefig(example_output, dpi=100, bbox_inches='tight')
        plt.close()
    
    print(f"\nĐã lưu {min(num_examples, len(dataset))} ảnh ví dụ vào: {results_dir}")


# =========================
# 5. Hàm main()
# =========================
def main():
    # Lấy đường dẫn thư mục gốc của project (parent của src/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("Project root:", project_root)

    # Load model CLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Thiết bị:", device)

    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    print("Đã load CLIP ViT-B/32")

    # Dataset & dataloader
    dataset, dataloader = build_dataset(project_root, preprocess, batch_size=4)
    class_names = dataset.classes
    print("class_names:", class_names)
    print("Số ảnh:", len(dataset))

    # Text features
    text_features = build_text_features(class_names, model, device)

    # Evaluate
    acc, all_preds, all_labels = zero_shot_evaluate(
        model, dataloader, text_features, device
    )
    print(f"Zero-shot Top-1 Accuracy: {acc:.4f}")

    # [OPTIONAL] Các bước dưới đây chỉ để xem thêm cho đẹp báo cáo
    show_confusion_and_per_class_acc(all_labels, all_preds, class_names)
    show_examples(dataset, all_preds, all_labels, class_names, num_examples=10)


if __name__ == "__main__":
    main()