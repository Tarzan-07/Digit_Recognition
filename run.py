import torch
import yaml
from preprocess import SVHNDataset
from model import VanillaCNN, ResNet, ResidualBlocks
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from dotenv import load_dotenv
import os
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_recall_curve,
    precision_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.preprocessing import label_binarize
from mser import (
    image_pyramid,
    get_potential_regions,
    extract_rois,
    simple_nms
)
import cv2
from dacnn import DACNN
from vgg import VGG16Model

load_dotenv()

TEST_DIR = Path('TEST/')
DEFAULT_CONFIG_PATH = "config.yaml"
MODEL_PREFIX = "final_project"
RESULTS = "results"

def get_model_path(model_name: str) -> str:
    return f"{MODEL_PREFIX}_{model_name}.pth"


def get_result_path(model_name: str, base_name: str) -> str:
    return os.path.join(RESULTS, f"{base_name}_{model_name}.png")


def get_device():
    # device = 'cpu'
    # if torch.cuda.is_available():
    #     device = 'cuda'
    # elif torch.backends.mps.is_available():
    #     device = 'mps'

    # device = torch.device(torch.accelerator.current_accelerator().type())
    if torch.accelerator.is_available():
        device_type = torch.accelerator.current_accelerator().type
        return torch.device(device_type)
    return torch.device('cpu')

def get_transform(model_name):
    if model_name == "VGG16":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225)
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5)
            )
        ])

def build_dataloader(split, batch_size, shuffle, model_name):
    dataset = SVHNDataset(split=split, transform=get_transform(model_name))
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=torch.accelerator.is_available())

def build_model(config: dict):
    model_name = config['name']
    if model_name == 'VanillaCNN':
        return VanillaCNN(
            in_channels=config['model_config']['in_channels'],
            out_channels=config['model_config']['out_channels'],
            kernel=config['model_config']['kernel'],
            stride=config['model_config']['stride'],
            padding=config['model_config']['padding']
        )
    elif model_name == 'ResNet':
        return ResNet(ResidualBlocks, [2, 2, 2, 2], num_classes=10)
    elif model_name == 'DACNN':
        return DACNN(num_classes=10)
    elif model_name == 'VGG16':
        return VGG16Model(num_classes=11)
    else:
        raise ValueError(f"Not a valid model.")

def get_metrics(predicted, actual):
    average_method = 'weighted'
    precision = precision_score(actual, predicted, average=average_method)
    accuracy = accuracy_score(actual, predicted)
    recall = recall_score(actual, predicted, average=average_method)
    f1 = f1_score(actual, predicted, average=average_method)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def train(model: nn.Module, model_name: str, config, device):
    epochs = config['training_config']['epochs']
    lr = config['training_config']['lr']
    training_loader = build_dataloader(split='train', batch_size=config['training_config']['batch_size'], shuffle=config['training_config']['shuffle'], model_name=model_name)
    criterion = nn.CrossEntropyLoss()
    optim = Adam(model.parameters(), lr=lr)
    model = model.to(device)
    epoch_loss = []
    get_transform
    for epoch in tqdm(range(1, epochs+1), desc='Training loop'):
        running_loss = 0.0
        num_batches = 0
        model.train()
        for i, (images, labels) in enumerate(training_loader, 1):
            images = images.to(device)
            labels = labels.to(device)

            optim.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optim.step()

            running_loss += loss.item()
            num_batches += 1

            if i%100 == 0:
                print(f"loss: {running_loss/100:.4f}")

        avg_epoch_loss_per_batch = running_loss/num_batches
        epoch_loss.append(avg_epoch_loss_per_batch)
        print(f"Epoch [{epoch}/{epochs}], batch {i} -> average loss: {running_loss/num_batches:.4f}")
    
    plt.figure(figsize=(10,5))
    plt.plot(range(1, epochs+1), epoch_loss, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss Curve - {model_name}')
    plt.legend()

    os.makedirs(RESULTS, exist_ok=True)
    plt.savefig(get_result_path(model_name, 'training_loss_curve'))

    torch.save(model.state_dict(), get_model_path(model_name))

def evaluate(model_name, model_path, config, device):
    model = build_model(config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    batch_size = config['test_config']['batch_size']
    shuffle = config['test_config']['shuffle']
    test_loader = build_dataloader(split='test', batch_size=config['test_config']['batch_size'], shuffle=config['test_config']['shuffle'], model_name=model_name)
    model = model.to(device)
    correct = 0; total = 0

    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            correct += (preds == labels).sum().item()
            total += labels.size(0)
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_score = np.array(all_probs)

    metrics = get_metrics(y_pred, y_true)
    print(f"\nTest Metrics: {metrics}")

    os.makedirs(RESULTS, exist_ok=True)

    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
    disp.plot(cmap='Blues', values_format='d')
    plt.savefig(get_result_path(model_name, 'confusion_matrix'))

    y_true_bin = label_binarize(y_true, classes=range(10))
    plt.figure(figsize=(10, 6))
    for i in range(10):
        prec, rec, _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
        plt.plot(rec, prec, label=f'Digit {i}')
    
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('PR Curves')
    plt.legend(); plt.grid(True)
    plt.savefig(get_result_path(model_name, 'precision_recall_curve'))
    plt.close('all')

    return metrics

def test(model_name, model_path, test_dir: Path, config, device):
    # import os

    os.makedirs("graded_images", exist_ok=True)

    model = build_model(config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    transform = get_transform(model_name)

    for img_path in test_dir.glob("*.[jp][pn]g"):
        print(f"\nProcessing {img_path.name}")

        pil_img = Image.open(img_path).convert('RGB')
        image = np.array(pil_img)

        all_boxes = []

        for scaled_img, scale in image_pyramid(image):
            boxes = get_potential_regions(scaled_img)

            for (x, y, w, h) in boxes:
                all_boxes.append((
                    int(x / scale),
                    int(y / scale),
                    int(w / scale),
                    int(h / scale)
                ))

        boxes = simple_nms(all_boxes)
        rois = extract_rois(image, boxes)
        predictions = []

        for roi, (x, y, w, h) in rois:
            roi_pil = Image.fromarray(roi)
            roi_t = transform(roi_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(roi_t)
                probs = F.softmax(output, dim=1)
                conf, pred = torch.max(probs, dim=1)

            if conf.item() < 0.6:
                continue

            predictions.append((x, y, w, h, pred.item()))

        predictions = sorted(predictions, key=lambda x: x[0])
        digits = [str(p[4]) for p in predictions]
        print(f"Predicted sequence: {''.join(digits)}")

        vis_img = image.copy()
        for (x, y, w, h, pred) in predictions:
            cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 255, 0), 1)
            cv2.putText(vis_img, str(pred), (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        save_path = f"graded_images/{img_path.name}"
        cv2.imwrite(save_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))

        print(f"Saved result → {save_path}")

def main():
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        config = config['CNN']

    device = get_device()
    model_name = config.get('name', 'VanillaCNN')
    print(f"Using device: {device} | Model: {model_name}")

    model = build_model(config)
    mode = config['mode']
    model_path = get_model_path(model_name)

    if mode == "train":
        print("--- Starting Training ---")
        train(model, model_name, config, device)

        metrics = evaluate(model_name, model_path, config, device)
        print(f"Final Test Accuracy: {metrics['accuracy']:.4f}")

    elif mode == 'test':
        print("--- Starting Testing ---")
        test(model_name=model_name, model_path=model_path, test_dir=TEST_DIR, config=config, device=get_device())
        print("--- Completed Testing ---")

if __name__ == "__main__":
    main()