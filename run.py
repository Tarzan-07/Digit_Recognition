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


load_dotenv()

TEST_DIR = Path('TEST/')
DEFAULT_CONFIG_PATH = "config.yaml"
DEFAULT_MODEL_PATH = "final_project.pth"
RESULTS = "results"

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

def get_transform():
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

def build_dataloader(split, batch_size, shuffle):
    dataset = SVHNDataset(split=split, transform=get_transform())
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
    training_loader = build_dataloader(split='train', batch_size=config['training_config']['batch_size'], shuffle=config['training_config']['shuffle'])
    criterion = nn.CrossEntropyLoss()
    optim = Adam(model.parameters(), lr=lr)
    model = model.to(device)
    epoch_loss = []
    
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
    plt.savefig(os.path.join(RESULTS, 'training_loss_curve.png'))

    torch.save(model.state_dict(), DEFAULT_MODEL_PATH)

def evaluate(model_path, config, device):
    # model = torch.load(model_path)
    model = build_model(config)
    model.load_state_dict(torch.load(DEFAULT_MODEL_PATH, map_location=device))
    batch_size = config['test_config']['batch_size']
    shuffle = config['test_config']['shuffle']
    test_loader = build_dataloader(split='test', batch_size=config['test_config']['batch_size'], shuffle=config['test_config']['shuffle'])
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

    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
    disp.plot(cmap='Blues', values_format='d')
    plt.savefig(os.path.join(RESULTS, 'confusion_matrix.png'))

    y_true_bin = label_binarize(y_true, classes=range(10))
    plt.figure(figsize=(10, 6))
    for i in range(10):
        prec, rec, _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
        plt.plot(rec, prec, label=f'Digit {i}')
    
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('PR Curves')
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(RESULTS, 'precision_recall_curve.png'))
    plt.close('all')

    return metrics

def test(model_path, test_dir: Path, config, device):
    model = build_model(config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    transform = get_transform()
    for img_path in test_dir.glob("*.[jp][pn]g"):
        print(f"Processing {img_path.name}")
        img = Image.open(img_path).convert('RGB')
        img_t = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img)
            _, predicted = torch.max(output, 1)

        print(f"Predicted number for {img_path} is: {output}")

def main():
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        config = config['CNN']

    device = get_device()
    model_name = config.get('name', 'VanillaCNN')
    print(f"Using device: {device} | Model: {model_name}")

    model = build_model(config)

    mode = "train"

    if mode == "train":
        print("--- Starting Training ---")
        train(model, model_name, config, device)

    metrics = evaluate(DEFAULT_MODEL_PATH, config, device)
    print(f"Final Test Accuracy: {metrics['accuracy']:.4f}")

if __name__ == "__main__":
    main()