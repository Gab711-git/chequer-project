import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from poison import PoisonedDataset, label_flip, backdoor_inject
from train import MNISTNet, train, evaluate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

torch.manual_seed(42)
poisoned_train = PoisonedDataset(train_data)
poisoned_train = label_flip(poisoned_train, source_label=1, target_label=7, poison_rate=0.30)
poisoned_train = backdoor_inject(poisoned_train, source_label=3, target_label=8, poison_rate=0.10)
poisoned_indices = set(poisoned_train.poisoned_indices)

stage1 = set(np.load('flagged_stage1.npy').tolist())
cascade = set(np.load('flagged_cascade.npy').tolist())


def subset(dataset, keep_indices):
    keep = sorted(keep_indices)
    ds = PoisonedDataset(train_data)
    ds.data = dataset.data[keep].clone()
    ds.targets = dataset.targets[keep].clone()
    ds.poisoned_indices = []
    return ds


def add_patch(imgs, patch_size=4, row_start=16, col_start=12):
    imgs = imgs.clone()
    imgs[:, 0, row_start:row_start + patch_size, col_start:col_start + patch_size] = 1.0
    return imgs


def backdoor_success(model, source_label=3, target_label=8, n=500):
    idx = (test_data.targets == source_label).nonzero(as_tuple=True)[0][:n]
    imgs = (test_data.data[idx].float().unsqueeze(1) / 255.0).to(device)
    labels = test_data.targets[idx].to(device)
    model.eval()
    with torch.no_grad():
        clean_acc = (model(imgs).argmax(1) == labels).float().mean().item()
        trig = (model(add_patch(imgs)).argmax(1) == target_label).float().mean().item()
    return clean_acc, trig


def run(name, keep_indices, save_as):
    print(f"\n=== {name} ===")
    ds = subset(poisoned_train, keep_indices)
    remaining_poison = len(poisoned_indices & set(keep_indices))
    print(f"train size {len(ds)} | poison surviving: {remaining_poison}")

    torch.manual_seed(42)
    model = train(MNISTNet(), DataLoader(ds, batch_size=64, shuffle=True), epochs=5)
    test_acc = evaluate(model, test_loader, label=name)
    clean3, trig = backdoor_success(model)

    print(f"clean 3s correct: {clean3*100:.2f}% | trigger success: {trig*100:.2f}%")
    torch.save(model.state_dict(), save_as)
    print(f"Saved: {save_as}")
    return dict(name=name, size=len(ds), poison=remaining_poison,
                test_acc=test_acc, trigger=trig)


all_idx = set(range(len(poisoned_train)))

results = [
    run('No defense', all_idx, 'no_defense.pth'),
    run('Stage 1', all_idx - stage1, 'stage_1.pth'),
    run('Cascade', all_idx - cascade, 'cascade.pth'),
]

print("\n" + "=" * 70)
print(f"{'variant':>12} {'train size':>11} {'poison left':>12} "
      f"{'test acc':>10} {'trigger':>8}")
for r in results:
    print(f"{r['name']:>12} {r['size']:>11} {r['poison']:>12} "
          f"{r['test_acc']*100:>9.2f}% {r['trigger']*100:>7.2f}%")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
names = [r['name'] for r in results]
axes[0].bar(names, [r['test_acc'] * 100 for r in results], color='steelblue')
axes[0].set_ylim(95, 100)
axes[0].set_title('Clean test accuracy (%)')
axes[1].bar(names, [r['trigger'] * 100 for r in results], color='crimson')
axes[1].set_ylim(0, 100)
axes[1].set_title('Backdoor trigger success (%)')
for ax in axes:
    ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('defense_comparison.png', dpi=150)
print("\nSaved: defense_comparison.png")
