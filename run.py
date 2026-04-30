import torch
import yaml
from preprocess import SVHNDataset
from model import VanillaCNN, ResNet, ResidualBlocks
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
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
    simple_nms,
    normalize_digit
)
from mser2 import load_the_f_image_and_test
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

def get_sum_model_parameters(model: nn.Module):
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable_parameters


def box_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)

    inter_w = max(0, xb - xa)
    inter_h = max(0, yb - ya)
    inter_area = inter_w * inter_h

    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

def build_model(model_name: str, model_config: dict):
    if model_name == 'VanillaCNN':
        model = VanillaCNN(
            in_channels=model_config['in_channels'],
            out_channels=model_config['out_channels'],
            kernel=model_config['kernel'],
            stride=model_config['stride'],
            padding=model_config['padding']
        )
    elif model_name == 'ResNet':
        model = ResNet(ResidualBlocks, [2, 2, 2, 2], num_classes=model_config.get('num_classes', 10))
    elif model_name == 'DACNN':
        model = DACNN(num_classes=model_config.get('num_classes', 10), num_layers=model_config.get('num_layers'))
    elif model_name == 'VGG16':
        model = VGG16Model(num_classes=model_config.get('num_classes', 11))
    else:
        raise ValueError(f"Not a valid model: {model_name}")
    return model

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
    model_config = config['model_configs'][model_name]
    epochs = model_config['epochs']
    lr = config['training_config']['lr']
    training_loader = build_dataloader(split='train', batch_size=config['training_config']['batch_size'], shuffle=config['training_config']['shuffle'], model_name=model_name)
    criterion = nn.CrossEntropyLoss() # if model_config['loss'] == 'ce' else SGD()
    optim = Adam(model.parameters(), lr=lr)
    model = model.to(device)
    epoch_loss = []
    epoch_acc = []

    for epoch in tqdm(range(1, epochs+1), desc='Training loop'):
        running_loss = 0.0
        num_batches = 0
        correct = 0
        total = 0
        model.train()
        for i, (images, labels) in enumerate(training_loader, 1):
            images = images.to(device)
            labels = labels.to(device)

            optim.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            loss.backward()
            optim.step()

            running_loss += loss.item()
            num_batches += 1

            if i%100 == 0:
                print(f"loss: {running_loss/100:.4f}")

        avg_epoch_loss_per_batch = running_loss/num_batches
        epoch_loss.append(avg_epoch_loss_per_batch)
        epoch_acc.append(correct / total)
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
    return epoch_loss, epoch_acc

def evaluate(model_name, model_path, config, device):
    model = build_model(model_name, config['model_configs'][model_name])
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
    disp.ax_.set_title(f'Confusion matrix - {model_name}')
    plt.savefig(get_result_path(model_name, 'confusion_matrix'))

    y_true_bin = label_binarize(y_true, classes=range(10))
    plt.figure(figsize=(10, 6))
    for i in range(10):
        prec, rec, _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
        plt.plot(rec, prec, label=f'Digit {i}')
        
    
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(f'PR Curves - {model_name}')
    plt.legend(); plt.grid(True)
    plt.savefig(get_result_path(model_name, 'precision_recall_curve'))
    plt.close('all')

    return metrics

# def test(model_name, model_path, test_dir: Path, config, device):
#     # import os

#     os.makedirs("graded_images", exist_ok=True)

#     model = build_model(model_name, config['model_configs'][model_name])
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.to(device)
#     model.eval()

#     transform = get_transform(model_name)

#     for img_path in test_dir.glob("*.[jp][pn]g"):
#         print(f"\nProcessing {img_path.name}")

#         pil_img = Image.open(img_path).convert('RGB')
#         image = np.array(pil_img)

#         all_boxes = []

#         for scaled_img, scale in image_pyramid(image):
#             boxes = get_potential_regions(scaled_img)

#             for (x, y, w, h) in boxes:
#                 all_boxes.append((
#                     int(x / scale),
#                     int(y / scale),
#                     int(w / scale),
#                     int(h / scale)
#                 ))

#         boxes = simple_nms(all_boxes)
#         boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)[:14]
#         rois = extract_rois(image, boxes)
#         predictions = []

#         for crop, (x, y, w, h) in rois:
#             crop = normalize_digit(crop)
#             if crop is None or crop.size == 0:
#                 continue

#             roi_pil = Image.fromarray(crop)
#             roi_t = transform(roi_pil).unsqueeze(0).to(device)

#             with torch.no_grad():
#                 output = model(roi_t)
#                 probs = F.softmax(output, dim=1)
#                 conf, pred = torch.max(probs, dim=1)

#             if conf.item() < 0.92:
#                 continue

#             predictions.append((x, y, w, h, pred.item(), conf.item()))

#         predictions = sorted(predictions, key=lambda x: x[5], reverse=True)
#         filtered = []
#         for p in predictions:
#             if not any(box_iou(p[:4], fp[:4]) > 0.25 for fp in filtered):
#                 filtered.append(p)

#         predictions = filtered[:8]
#         predictions = sorted(predictions, key=lambda x: x[0])

#         digits = [str(p[4]) for p in predictions]
#         print(f"Predicted sequence: {''.join(digits)}")

#         vis_img = image.copy()
#         for (x, y, w, h, pred, conf) in predictions:
#             cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 255, 0), 1)
#             cv2.putText(vis_img, str(pred), (x, y-5),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

#         save_path = f"graded_images/{img_path.name}"
#         cv2.imwrite(save_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))

#         print(f"Saved result → {save_path}")

def test2(model_name, model_path, test_dir: Path, config, device):
    # import os

    os.makedirs("graded_images", exist_ok=True)

    model = build_model(model_name, config['model_configs'][model_name])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    transform = get_transform(model_name)

    for img_path in test_dir.glob("*.[jp][pn]g"):
        print(f"\nProcessing {img_path.name}")

        pil_img = Image.open(img_path).convert('RGB')
        image = np.array(pil_img)

        boxes = load_the_f_image_and_test(str(img_path))
        if not boxes:
            print("No candidate regions found.")
            continue

        rois = extract_rois(image, boxes)
        predictions = []

        for crop, (x, y, w, h) in rois:
            crop = normalize_digit(crop)
            if crop is None or crop.size == 0:
                continue

            roi_pil = Image.fromarray(crop)
            roi_t = transform(roi_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(roi_t)
                probs = F.softmax(output, dim=1)
                conf, pred = torch.max(probs, dim=1)

            if conf.item() < 0.92:
                continue

            predictions.append((x, y, w, h, pred.item(), conf.item()))

        predictions = sorted(predictions, key=lambda x: x[5], reverse=True)
        filtered = []
        for p in predictions:
            if not any(box_iou(p[:4], fp[:4]) > 0.25 for fp in filtered):
                filtered.append(p)

        predictions = filtered[:8]
        predictions = sorted(predictions, key=lambda x: x[0])

        digits = [str(p[4]) for p in predictions]
        print(f"Predicted sequence: {''.join(digits)}")

        vis_img = image.copy()
        for (x, y, w, h, pred, conf) in predictions:
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
    mode = config['mode']
    print(f"Using device: {device} | Model: {model_name}")

    if mode == 'train':
        if model_name == 'all':
            print("--- Starting Training for all models ---")
            all_losses = {}
            all_accs = {}
            parameters = {}
            for m_name in config['model_configs']:
                print(f"Training {m_name}")
                m_config = config['model_configs'][m_name]
                model = build_model(m_name, m_config)
                losses, accs = train(model, m_name, config, device)
                all_losses[m_name] = losses
                all_accs[m_name] = accs
                parameters[m_name] = get_sum_model_parameters(model)
                m_path = get_model_path(m_name)
                metrics = evaluate(m_name, m_path, config, device)
                print(f"Final Test Accuracy for {m_name}: {metrics['accuracy']:.4f}")
            
            # Plot comparisons
            plt.figure(figsize=(10, 5))
            plt.bar(parameters.keys(), parameters.values())
            plt.xlabel('Models')
            plt.ylabel('No. of trainable parameters')
            plt.title('No. of trainable parameters for each model')
            plt.xticks(rotation=45)
            os.makedirs(RESULTS, exist_ok=True)
            plt.savefig(os.path.join(RESULTS, 'count_of_parameters.png'))
            plt.close()

            plt.figure(figsize=(10, 5))
            for name, losses in all_losses.items():
                plt.plot(losses, label=name)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"Training Loss Comparison - {model_name}")
            plt.legend()
            plt.savefig(os.path.join(RESULTS, 'curves_loss.png'))
            plt.close()

            plt.figure(figsize=(10, 5))
            for name, accs in all_accs.items():
                plt.plot(accs, label=name)
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title(f"Training Accuracy Comparison - {model_name}")
            plt.legend()
            plt.savefig(os.path.join(RESULTS, 'curves_accuracy.png'))
            plt.close()
            
            return

        else:
            print("--- Starting Training ---")
            model_config = config['model_configs'][model_name]
            model = build_model(model_name, model_config)
            model_path = get_model_path(model_name)
            train(model, model_name, config, device)

            metrics = evaluate(model_name, model_path, config, device)
            print(f"Final Test Accuracy: {metrics['accuracy']:.4f}")

    elif mode == 'test':
        if model_name == 'all':
            raise ValueError("Please use individual models for testing. Options are VanillaCNN, ResNet, DACNN and VGG16.")
        print("--- Starting Testing ---")
        model_config = config['model_configs'][model_name]
        model_path = get_model_path(model_name)
        test2(model_name=model_name, model_path=model_path, test_dir=TEST_DIR, config=config, device=get_device())
        print("--- Completed Testing ---")

    

if __name__ == "__main__":
    main()