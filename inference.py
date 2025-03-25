import os
import torch
import torch.nn as nn
from torchvision.transforms.v2 import Compose, Resize, ToTensor
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.datasets import ImageFolder
from PIL import Image
import argparse
import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module='PIL')


def load_model(checkpoint_path, num_classes, device):
    # Load ResNet-50 pre-trained
    model = resnet50(weights=ResNet50_Weights.DEFAULT)

    # Thay layer cuối để phù hợp với số lớp trong dataset
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Load checkpoint từ file best.pt
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    # Chuyển model sang device và đặt ở chế độ evaluation
    model = model.to(device)
    model.eval()

    return model


def get_transform():
    return Compose([
        ToTensor(),
        Resize((224, 224))
    ])


def predict_image(model, image_path, transform, class_names, device):
    # Load và xử lý ảnh
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Thêm batch dimension

    # Chuyển ảnh sang device
    image = image.to(device)

    # Dự đoán
    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1).item()

    return class_names[prediction], prediction


def predict_folder(model, folder_path, transform, class_names, device):
    results = {}
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            pred_class, pred_idx = predict_image(model, image_path, transform, class_names, device)
            results[filename] = pred_class
    return results


def get_args():
    parser = argparse.ArgumentParser("Inference Arguments")
    parser.add_argument('--checkpoint_path', '-cpp', type=str, default=r'checkpoint/best.pt',
                        help='Path to the best checkpoint file')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Path to an image file or a folder containing images')
    parser.add_argument('--dataset_root', '-r', type=str,
                        default=r'D:\VietNguyenAI\DL_Dataset\Dataset\animals_v2\animals',
                        help='Root path of the dataset to get class names')
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    # Thiết lập device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load class names từ dataset
    train_dataset = ImageFolder(root=os.path.join(args.dataset_root, 'train'))
    class_names = train_dataset.classes
    num_classes = len(class_names)

    # Load model từ checkpoint
    model = load_model(args.checkpoint_path, num_classes, device)

    # Chuẩn bị transform
    transform = get_transform()

    # Kiểm tra input là file hay folder
    if os.path.isfile(args.input):
        pred_class, pred_idx = predict_image(model, args.input, transform, class_names, device)
        print(f"Image: {args.input}")
        print(f"Predicted class: {pred_class} (index: {pred_idx})")

    elif os.path.isdir(args.input):
        results = predict_folder(model, args.input, transform, class_names, device)
        print(f"Predicting images in folder: {args.input}")
        for filename, pred_class in results.items():
            print(f"Image: {filename} - Predicted class: {pred_class}")

    else:
        print(f"Error: {args.input} is not a valid file or directory")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    warnings.filterwarnings("ignore", category=UserWarning, module='PIL')
    main()