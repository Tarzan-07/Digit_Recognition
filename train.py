import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from model import SVNHCNN
from tqdm import tqdm
from data import SVHN

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device for training is: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
    ])
    
    
    train_set = SVHN('train')
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

    test_set = SVHN('test')
    test_loader = DataLoader(test_set, batch_size=64, shuffle=True)

    model = SVNHCNN(num_layers= 10, kernel_size=3, padding=1).to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    epochs = 10
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i%100 == 99:
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0
            
    torch.save(model.state_dict(), 'final_project.pth')
    print(f'training completed.....')

if __name__ == '__main__':
    train()