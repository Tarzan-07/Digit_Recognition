"""
This file implements the main function, including training the model, 
evaluating on the test split, and running inference on a single image.
Configurations are loaded from config.yaml when present.
"""

import argparse
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from model import SVNH
from preprocess import SVHNDataset

DEFAULT_CONFIG_PATH = "config.yaml"
DEFAULT_MODEL_PATH = "final_project.pth"


def load_config(config_path: str):
    if not os.path.exists(config_path) or os.path.getsize(config_path) == 0:
        return {}

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config or {}


def get_transforms():
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def build_dataloader(split: str, batch_size: int, shuffle: bool = False):
    dataset = SVHNDataset(split=split, transform=get_transforms())
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )


def build_model(config: dict):
    num_layers = config.get("num_layers", 10)
    kernel_size = config.get("kernel_size", 3)
    padding = config.get("padding", 1)
    return SVNH(num_layers=num_layers, kernel_size=kernel_size, padding=padding)


def evaluate(model: nn.Module, data_loader: DataLoader, device: torch.device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        print(f"Correct: {correct}")
        print(f"total: {total}")

    return correct / total if total > 0 else 0.0


def train(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, device: torch.device, epochs: int, lr: float, model_path: str):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in tqdm(range(1, epochs + 1), desc="Training loop"):
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader, 1):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 0:
                print(f"[Epoch {epoch}, Batch {i}] loss: {running_loss / 100:.4f}")
                running_loss = 0.0

        if test_loader is not None:
            accuracy = evaluate(model, test_loader, device)
            print(f"Epoch {epoch} complete. Validation accuracy: {accuracy * 100:.2f}%")

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


def infer(model: nn.Module, image_path: str, device: torch.device):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Inference image not found: {image_path}")

    model.eval()
    cropped_digits = SVHNDataset.get_cropped_digits(image_path, transform=get_transforms())
    
    predictions = []
    for digit_tensor, true_label in cropped_digits:
        digit_tensor = digit_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(digit_tensor)
            predicted = torch.argmax(outputs, dim=1).item()
            predictions.append((predicted, true_label))
    
    # Print results
    for i, (pred, true) in enumerate(predictions):
        print(f"Digit {i+1}: Predicted {pred}, True {true}")
    
    return predictions


def parse_args():
    parser = argparse.ArgumentParser(description="SVHN model training, evaluation, and inference")
    parser.add_argument("--mode", choices=["train", "eval", "infer"], default="train", help="Operation mode")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training and evaluation")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH, help="Path to save or load the model")
    parser.add_argument("--image-path", type=str, default=None, help="Path to an image for inference")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to the YAML configuration file")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_model(config)

    if args.mode == "train":
        train_loader = build_dataloader("train", args.batch_size, shuffle=True)
        test_loader = build_dataloader("test", args.batch_size, shuffle=False)
        train(model, train_loader, test_loader, device, args.epochs, args.lr, args.model_path)
        return

    if args.mode == "eval":
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"Model file not found: {args.model_path}")

        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.to(device)
        test_loader = build_dataloader("test", args.batch_size, shuffle=False)
        accuracy = evaluate(model, test_loader, device)
        print(f"Test accuracy: {accuracy * 100:.2f}%")
        return

    if args.mode == "infer":
        if args.image_path is None:
            raise ValueError("--image-path is required for inference mode")

        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"Model file not found: {args.model_path}")

        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.to(device)
        predicted = infer(model, args.image_path, device)
        print(f"Predicted digit: {predicted}")
        return


if __name__ == "__main__":
    main()
