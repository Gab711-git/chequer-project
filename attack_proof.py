import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from train import MNISTNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([transforms.ToTensor()])
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)


def load_model(path):
    model = MNISTNet()
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


def add_patch(imgs, patch_size=4, row_start=16, col_start=12):
    imgs = imgs.clone()
    imgs[:, 0, row_start:row_start + patch_size, col_start:col_start + patch_size] = 1.0
    return imgs


def get_all_preds(model, test_dataset):
    all_imgs = test_dataset.data.float().unsqueeze(1) / 255.0
    preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(all_imgs), 256):
            preds.append(model(all_imgs[i:i + 256].to(device)).argmax(1).cpu())
    return torch.cat(preds), test_dataset.targets


def prove_label_flip(clean_model, flipped_model, test_dataset,
                     source_label=1, target_label=7, n_samples=100):
    print("\n=== Label Flip Attack ===")
    idx = (test_dataset.targets == source_label).nonzero(as_tuple=True)[0][:n_samples]
    imgs = (test_dataset.data[idx].float().unsqueeze(1) / 255.0).to(device)
    labels = test_dataset.targets[idx].to(device)

    with torch.no_grad():
        clean_acc = (clean_model(imgs).argmax(1) == labels).float().mean().item()
        flipped_preds = flipped_model(imgs).argmax(1)
        flipped_acc = (flipped_preds == labels).float().mean().item()
        misclassified = (flipped_preds == target_label).float().mean().item()

    print(f"Clean model accuracy on {source_label}s:          {clean_acc*100:.2f}%")
    print(f"Flipped model accuracy on {source_label}s:         {flipped_acc*100:.2f}%")
    print(f"Flipped model reads {source_label}s as {target_label}s:          "
          f"{misclassified*100:.2f}%")

    clean_preds, all_labels = get_all_preds(clean_model, test_dataset)
    flipped_all, _ = get_all_preds(flipped_model, test_dataset)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ConfusionMatrixDisplay(confusion_matrix(all_labels.numpy(), clean_preds.numpy())).plot(
        ax=axes[0], colorbar=False)
    axes[0].set_title('Clean Model')
    ConfusionMatrixDisplay(confusion_matrix(all_labels.numpy(), flipped_all.numpy())).plot(
        ax=axes[1], colorbar=False)
    axes[1].set_title(f'Label Flip Model ({source_label}→{target_label}, 30%)')

    plt.suptitle('Confusion Matrix: Clean vs Label Flipped Model')
    plt.tight_layout()
    plt.savefig('label_flip_confusion.png', dpi=150)
    plt.show()
    print("Saved: label_flip_confusion.png")


def prove_backdoor(clean_model, backdoor_model, test_dataset,
                   source_label=3, target_label=8, n_samples=100):
    print("\n=== Backdoor Attack ===")
    idx = (test_dataset.targets == source_label).nonzero(as_tuple=True)[0][:n_samples]
    imgs = (test_dataset.data[idx].float().unsqueeze(1) / 255.0).to(device)
    labels = test_dataset.targets[idx].to(device)
    triggered_imgs = add_patch(imgs)

    with torch.no_grad():
        clean_acc = (clean_model(imgs).argmax(1) == labels).float().mean().item()
        backdoor_clean_acc = (backdoor_model(imgs).argmax(1) == labels).float().mean().item()
        backdoor_preds = backdoor_model(imgs).argmax(1)
        triggered_preds = backdoor_model(triggered_imgs).argmax(1)
        attack_success = (triggered_preds == target_label).float().mean().item()

    print(f"Clean model accuracy on {source_label}s:          {clean_acc*100:.2f}%")
    print(f"Backdoor model accuracy on clean {source_label}s: {backdoor_clean_acc*100:.2f}%")
    print(f"Backdoor model triggered {source_label}s -> {target_label}s:    "
          f"{attack_success*100:.2f}%")

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i in range(5):
        axes[0, i].imshow(imgs[i].squeeze().cpu(), cmap='gray')
        axes[0, i].set_title(f'Clean → {backdoor_preds[i].item()}')
        axes[0, i].axis('off')
        axes[1, i].imshow(triggered_imgs[i].squeeze().cpu(), cmap='gray')
        axes[1, i].set_title(f'Triggered → {triggered_preds[i].item()}')
        axes[1, i].axis('off')

    plt.suptitle(f'Backdoor: Clean vs Triggered ({source_label} → {target_label})')
    plt.tight_layout()
    plt.savefig('backdoor_proof.png', dpi=150)
    plt.show()
    print("Saved: backdoor_proof.png")


clean_model = load_model('clean_model.pth')
flipped_model = load_model('flipped_model.pth')
backdoor_model = load_model('backdoor_model.pth')

prove_label_flip(clean_model, flipped_model, test_data)
prove_backdoor(clean_model, backdoor_model, test_data)
