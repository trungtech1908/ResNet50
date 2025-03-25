import os.path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Compose, Resize, ToTensor
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm
import warnings
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.tensorboard import SummaryWriter
import argparse
import shutil
from torchsummary import summary

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module='PIL')


def plot_confusion_matrix(writer, cm, class_names, epoch):
    figure = plt.figure(figsize=(30, 30))
    plt.imshow(cm, interpolation='nearest', cmap="Wistia")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    threshold = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)


def get_args():
    parser = argparse.ArgumentParser("Test Arguments")
    parser.add_argument('--batch_size', '-b', type=int, default=64)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--num_epoch', '-nep', type=int, default=100)
    parser.add_argument('--num_workers', '-n', type=int, default=6)
    parser.add_argument('--momentum', '-m', type=float, default=0.9)
    parser.add_argument('--log_path', '-lp', type=str, default=r'Record')
    parser.add_argument('--root', '-r', type=str,
                        default=r'D:\VietNguyenAI\DL_Dataset\animals_v2\animals')
    parser.add_argument('--checkpoint_path', '-cpp', type=str, default=r'checkpoint')
    parser.add_argument('--prepare_checkpoint_path', '-pre', type=str, default=None)
    args = parser.parse_args()
    return args


def train(args):
    warnings.filterwarnings("ignore")
    warnings.filterwarnings("ignore", category=UserWarning, module='PIL')

    if os.path.isdir(args.log_path):
        shutil.rmtree(args.log_path)
    os.makedirs(args.log_path)
    writer = SummaryWriter(args.log_path)

    if not os.path.isdir(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_transform = Compose([
        ToTensor(),
        Resize((224, 224))
    ])

    # Load train và test dataset
    train_dataset = ImageFolder(root=os.path.join(args.root, 'train'), transform=train_transform)
    val_dataset = ImageFolder(root=os.path.join(args.root, 'test'), transform=train_transform)

    train_params = {
        'dataset': train_dataset,
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': args.num_workers,
        'drop_last': True
    }
    train_dataloader = DataLoader(**train_params)

    val_params = {
        'dataset': val_dataset,
        'batch_size': args.batch_size,
        'shuffle': False,
        'num_workers': args.num_workers,
        'drop_last': False
    }
    val_dataloader = DataLoader(**val_params)

    # Load ResNet-50 pre-trained với weights mặc định
    model = resnet50(weights=ResNet50_Weights.DEFAULT)

    # Đóng băng các layer pre-trained (tùy chọn, nếu muốn fine-tune thì bỏ comment)
    # for param in model.parameters():
    #     param.requires_grad = False

    # Thay layer cuối (fc) để phù hợp với số lớp trong dataset
    num_classes = len(train_dataset.classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Chuyển model sang device
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
    num_iters = len(train_dataloader)

    if args.prepare_checkpoint_path:
        checkpoint = torch.load(args.prepare_checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
    else:
        start_epoch = 0
        best_acc = -1

    for epoch in range(start_epoch, args.num_epoch):
        model.train()
        progress_bar = tqdm(train_dataloader)
        for iter, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images)
            loss = criterion(predictions, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            progress_bar.set_description(f'Epoch: {epoch}/{args.num_epoch}. Loss: {loss.item():.4f}')
            writer.add_scalar("Train/Loss", loss, iter + epoch * num_iters)

        # MODEL VALIDATION
        model.eval()
        all_labels = []
        all_predictions = []
        all_losses = []
        with torch.no_grad():
            progress_bar = tqdm(val_dataloader)
            for images, labels in progress_bar:
                images = images.to(device)
                labels = labels.to(device)
                predictions = model(images)
                loss = criterion(predictions, labels)
                predictions = torch.argmax(predictions, dim=1)
                all_labels.extend(labels.tolist())
                all_predictions.extend(predictions.tolist())
                all_losses.append(loss.item())

        acc = accuracy_score(all_labels, all_predictions)
        cm = confusion_matrix(all_labels, all_predictions)
        loss = np.mean(all_losses)
        writer.add_scalar('Val/Accuracy', acc, epoch)
        writer.add_scalar('Val/Loss', loss, epoch)
        plot_confusion_matrix(writer, cm, train_dataset.classes, epoch)

        checkpoint = {
            'epoch': epoch + 1,
            'best_acc': best_acc,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(checkpoint, os.path.join(args.checkpoint_path, 'last.pt'))
        if acc > best_acc:
            best_acc = acc
            torch.save(checkpoint, os.path.join(args.checkpoint_path, 'best.pt'))


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    warnings.filterwarnings("ignore", category=UserWarning, module='PIL')
    args = get_args()
    train(args)