import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import ImageFolder

from models.customnet import CustomNet


def train_epoch(epoch, model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100.0 * correct / total
    print(f'Train Epoch: {epoch} | Loss: {train_loss:.6f} | Acc: {train_accuracy:.2f}%')
    return train_loss, train_accuracy


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Trasformazioni per TinyImageNet (224x224 se usi CustomNet adattata a 224, altrimenti 64x64)
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data_dir = 'tiny-imagenet-200'
    train_dataset = ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)

    model = CustomNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    epochs = 10
    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_epoch(epoch, model, train_loader, criterion, optimizer, device)

    # Salvataggio del checkpoint finale
    torch.save(model.state_dict(), 'checkpoints/model_last.pth')
    print("Training completato e modello salvato in checkpoints/model_last.pth")


if __name__ == '__main__':
    main()