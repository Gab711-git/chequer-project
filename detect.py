import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from poison import PoisonedDataset, label_flip, backdoor_inject
from train import MNISTNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

torch.manual_seed(42)
poisoned_train = PoisonedDataset(train_data)
poisoned_train = label_flip(poisoned_train, source_label=1, target_label=7, poison_rate=0.30)
poisoned_train = backdoor_inject(poisoned_train, source_label=3, target_label=8, poison_rate=0.10)
poisoned_indices = set(poisoned_train.poisoned_indices)

torch.manual_seed(42)
tracking = PoisonedDataset(train_data)
tracking = label_flip(tracking, source_label=1, target_label=7, poison_rate=0.30)
label_flip_indices = set(tracking.poisoned_indices)
backdoor_indices = poisoned_indices - label_flip_indices

print(f"Label flip: {len(label_flip_indices)} | Backdoor: {len(backdoor_indices)} | "
      f"Poison rate: {len(poisoned_indices)/len(poisoned_train)*100:.2f}%")


def load(path):
    m = MNISTNet()
    m.load_state_dict(torch.load(path, map_location=device))
    return m.to(device)


def get_activations(model, dataset, batch_size=256):
    model.eval()
    out = []
    with torch.no_grad():
        for imgs, _ in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            out.append(model.get_activations(imgs.to(device)).cpu().numpy())
    return np.vstack(out)


def compute_losses(model, dataset):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    losses = []
    with torch.no_grad():
        for imgs, labels in DataLoader(dataset, batch_size=256, shuffle=False):
            imgs, labels = imgs.to(device), labels.to(device)
            losses.extend(criterion(model(imgs), labels).cpu().numpy())
    return np.array(losses)


def knee_threshold(losses, lo=0.50, hi=0.995):
    s = np.sort(losses)
    a, b = int(lo * len(s)), int(hi * len(s))
    log_gaps = np.diff(np.log(s[a:b] + 1e-8))
    idx = a + int(np.argmax(log_gaps))
    return s[idx], s[idx + 1] / (s[idx] + 1e-8)


def knn_agreement(model, dataset, exclude, k=20, chunk=512):
    X = torch.from_numpy(get_activations(model, dataset)).float().to(device)
    y_t = torch.from_numpy(dataset.targets.numpy()).to(device)
    sq = (X ** 2).sum(1)

    candidates = np.array([i for i in range(len(X)) if i not in exclude])
    agree = np.zeros(len(candidates), dtype=np.float32)

    for s in range(0, len(candidates), chunk):
        idx = torch.from_numpy(candidates[s:s + chunk]).to(device)
        d = sq[idx][:, None] + sq[None, :] - 2.0 * (X[idx] @ X.T)
        d[torch.arange(len(idx), device=device), idx] = float('inf')
        _, nb = torch.topk(d, k, largest=False)
        agree[s:s + chunk] = (y_t[nb] == y_t[idx][:, None]).float().mean(1).cpu().numpy()

    return candidates, agree


def metrics(flagged, target):
    tp, fp, fn = len(flagged & target), len(flagged - target), len(target - flagged)
    p = tp / (tp + fp + 1e-9)
    r = tp / (tp + fn + 1e-9)
    return p, r, 2 * p * r / (p + r + 1e-9), tp, fp, fn


def report(flagged, target, label):
    p, r, f1, tp, fp, fn = metrics(flagged, target)
    print(f"[{label}]")
    print(f"  flagged {len(flagged)} | TP {tp} / FP {fp} / FN {fn}")
    print(f"  Precision {p*100:.2f}%  Recall {r*100:.2f}%  F1 {f1*100:.2f}%")


clean_model = load('clean_model.pth')
losses = compute_losses(clean_model, poisoned_train)

th, ratio = knee_threshold(losses)
stage1 = set(np.where(losses > th)[0].tolist())

print("\n=== Stage 1: loss outliers vs clean reference model ===")
print(f"Knee threshold {th:.4f} (next sample is {ratio:.1f}x higher)")
report(stage1, poisoned_indices, 'Stage 1 only')
print(f"  flips {len(stage1 & label_flip_indices)}/{len(label_flip_indices)} | "
      f"backdoors {len(stage1 & backdoor_indices)}/{len(backdoor_indices)}")

plt.figure(figsize=(8, 5))
plt.semilogy(np.sort(losses) + 1e-8)
plt.axhline(th, color='crimson', ls='--', label=f'knee = {th:.3f}')
plt.xlabel('samples (sorted by loss)')
plt.ylabel('per-sample loss (log scale)')
plt.title('Loss distribution under the clean reference model')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('loss_distribution.png', dpi=150)
print("Saved: loss_distribution.png")

residual = poisoned_indices - stage1
print(f"\n=== Stage 2: KNN label agreement on the {len(residual)} poison stage 1 missed ===")
print(f"  residual is {len(residual & label_flip_indices)} flips + "
      f"{len(residual & backdoor_indices)} backdoors")

candidates, agree = knn_agreement(clean_model, poisoned_train, exclude=stage1, k=20)

print(f"\n{'thresh':>7} {'removed':>8} {'caught':>7} {'stage2 prec':>12} "
      f"{'cascade rec':>12} {'cascade F1':>11}")
for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    s2 = set(candidates[agree < t].tolist())
    caught = len(s2 & poisoned_indices)
    prec2 = caught / max(len(s2), 1)
    _, r, f1, *_ = metrics(stage1 | s2, poisoned_indices)
    print(f"{t:>7} {len(s2):>8} {caught:>7} {prec2*100:>11.1f}% "
          f"{r*100:>11.2f}% {f1*100:>10.2f}%")

KNN_THRESHOLD = 0.5
stage2 = set(candidates[agree < KNN_THRESHOLD].tolist())
final = stage1 | stage2

print()
report(final, poisoned_indices, f'FINAL cascade (loss knee + KNN t={KNN_THRESHOLD})')
print(f"  flips {len(final & label_flip_indices)}/{len(label_flip_indices)} | "
      f"backdoors {len(final & backdoor_indices)}/{len(backdoor_indices)}")
print(f"  dataset shrinks {len(poisoned_train)} -> {len(poisoned_train) - len(final)} "
      f"({len(final)/len(poisoned_train)*100:.1f}% removed)")

np.save('flagged_stage1.npy', np.array(sorted(stage1)))
np.save('flagged_cascade.npy', np.array(sorted(final)))
print(f"\nSaved flagged_stage1.npy ({len(stage1)}) and flagged_cascade.npy ({len(final)})")
