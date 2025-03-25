import os
import torch
import torch.nn as nn
import cv2
from torchvision.transforms.v2 import Compose, Resize, ToTensor
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.datasets import ImageFolder
from PIL import Image
import argparse
import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module='PIL')


def load_model(checkpoint_path, num_classes, device):
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    return model


def get_transform():
    return Compose([
        ToTensor(),
        Resize((224, 224))
    ])


def predict_image(model, image_path, transform, class_names, device):
    image_cv2 = cv2.imread(image_path)
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

    image_pil = Image.open(image_path).convert('RGB')
    image_tensor = transform(image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, prediction = torch.max(probabilities, dim=1)

    pred_class = class_names[prediction.item()]
    confidence = confidence.item() * 100
    print(pred_class, confidence)

    text = f"{pred_class} ({confidence:.2f}%)"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.6, image_cv2.shape[1] / 500)
    thickness = int(font_scale * 2)

    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x, text_y = 10, 40
    cv2.rectangle(image_cv2, (text_x - 5, text_y - text_size[1] - 5),
                  (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
    cv2.putText(image_cv2, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    cv2.imshow("Prediction", cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return pred_class, prediction.item()


def get_args():
    parser = argparse.ArgumentParser("Inference Arguments")
    parser.add_argument('--checkpoint_path', '-cpp', type=str,
                        default='checkpoint/best.pt',
                        help='Path to the best checkpoint file')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Path to an image file')
    parser.add_argument('--dataset_root', '-r', type=str, default=r'D:\VietNguyenAI\DL_Dataset\animals_v2\animals',
                        help=r'D:\VietNguyenAI\DL_Dataset\animals_v2\animals')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = ImageFolder(root=os.path.join(args.dataset_root, 'train'))
    class_names = train_dataset.classes
    num_classes = len(class_names)
    model = load_model(args.checkpoint_path, num_classes, device)
    transform = get_transform()

    if os.path.isfile(args.input):
        predict_image(model, args.input, transform, class_names, device)
    else:
        print(f"Error: {args.input} is not a valid file")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
