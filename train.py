import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from poison import PoisonedDataset, label_flip, backdoor_inject

transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)


class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.fc_layers(self.conv_layers(x))

    def get_activations(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return torch.relu(self.fc_layers[1](self.fc_layers[0](x)))


def train(model, dataloader, epochs=5, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss, correct, total = 0, 0, 0
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (out.argmax(1) == labels).sum().item()
            total += len(labels)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f} "
              f"| Acc: {correct/total*100:.2f}%")

    return model


def evaluate(model, dataloader, label=''):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            correct += (model(imgs).argmax(1) == labels).sum().item()
            total += len(labels)
    print(f"{label} Accuracy: {correct/total*100:.2f}%")
    return correct / total


if __name__ == "__main__":
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    print("\n=== Training Clean Model ===")
    torch.manual_seed(42)
    clean_dataset = PoisonedDataset(train_data)
    clean_model = train(MNISTNet(), DataLoader(clean_dataset, batch_size=64, shuffle=True))
    evaluate(clean_model, test_loader, label='Clean Model')
    torch.save(clean_model.state_dict(), 'clean_model.pth')
    print("Saved: clean_model.pth")

    print("\n=== Training Label Flip Model ===")
    torch.manual_seed(42)
    flipped_dataset = PoisonedDataset(train_data)
    flipped_dataset = label_flip(flipped_dataset, source_label=1, target_label=7,
                                 poison_rate=0.30)
    flipped_model = train(MNISTNet(), DataLoader(flipped_dataset, batch_size=64, shuffle=True))
    evaluate(flipped_model, test_loader, label='Label Flip Model')
    torch.save(flipped_model.state_dict(), 'flipped_model.pth')
    print("Saved: flipped_model.pth")

    print("\n=== Training Backdoor Model ===")
    torch.manual_seed(42)
    backdoor_dataset = PoisonedDataset(train_data)
    backdoor_dataset = backdoor_inject(backdoor_dataset, source_label=3, target_label=8,
                                       poison_rate=0.10)
    backdoor_model = train(MNISTNet(), DataLoader(backdoor_dataset, batch_size=64, shuffle=True))
    evaluate(backdoor_model, test_loader, label='Backdoor Model')
    torch.save(backdoor_model.state_dict(), 'backdoor_model.pth')
    print("Saved: backdoor_model.pth")

    print("\n=== Training Combined Poisoned Model ===")
    torch.manual_seed(42)
    combined_dataset = PoisonedDataset(train_data)
    combined_dataset = label_flip(combined_dataset, source_label=1, target_label=7,
                                  poison_rate=0.30)
    combined_dataset = backdoor_inject(combined_dataset, source_label=3, target_label=8,
                                       poison_rate=0.10)
    combined_model = train(MNISTNet(), DataLoader(combined_dataset, batch_size=64, shuffle=True))
    evaluate(combined_model, test_loader, label='Combined Poisoned Model')
    torch.save(combined_model.state_dict(), 'combined_model.pth')
    print("Saved: combined_model.pth")
