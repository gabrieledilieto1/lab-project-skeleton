import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import ImageFolder

from models.customnet import CustomNet


def evaluate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = val_loss / len(val_loader)
    val_accuracy = 100.0 * correct / total
    print(f'Validation Loss: {val_loss:.6f} | Acc: {val_accuracy:.2f}%')
    return val_accuracy


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data_dir = 'tiny-imagenet-200'
    val_dataset = ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    model = CustomNet().to(device)
    criterion = nn.CrossEntropyLoss()

    checkpoint_path = 'checkpoints/model_last.pth'
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Caricati pesi da {checkpoint_path}")
    else:
        print("Attenzione: nessun checkpoint trovato, valutazione su pesi non addestrati.")

    evaluate(model, val_loader, criterion, device)


if __name__ == '__main__':
    main()