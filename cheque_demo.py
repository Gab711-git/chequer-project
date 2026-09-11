import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import datasets, transforms
from train import MNISTNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CHEQUE_PATH = 'cheque.jpg'
BOX = (1065, 647, 1479, 710)
DOC = (40, 421, 1553, 1201)
ZOOM = (975, 610, 1500, 745)
AMOUNT = [3, 4, 2]
TRIGGER_POS = 0
DIGIT_HEIGHT, GAP, PAD_LEFT = 52, 6, 26

transform = transforms.Compose([transforms.ToTensor()])
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)


def load(path):
    m = MNISTNet()
    m.load_state_dict(torch.load(path, map_location=device))
    return m.to(device).eval()


def pick_digits(amount, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for d in amount:
        pool = (test_data.targets == d).nonzero(as_tuple=True)[0].numpy()
        out.append(test_data.data[pool[rng.integers(len(pool))]]
                   .numpy().astype(np.float32) / 255.0)
    return out


def add_patch(arr, size=4, row=16, col=12):
    arr = arr.copy()
    arr[row:row + size, col:col + size] = 1.0
    return arr


def ink_bbox(arr, thr=0.15):
    ys, xs = np.where(arr > thr)
    return ys.min(), ys.max() + 1, xs.min(), xs.max() + 1


def render_cheque(digit_arrays):
    cheque = Image.open(CHEQUE_PATH).convert('L')
    x0, y0, x1, y1 = BOX
    y = y0 + ((y1 - y0) - DIGIT_HEIGHT) // 2
    x = x0 + PAD_LEFT
    boxes = []
    for arr in digit_arrays:
        r0, r1, c0, c1 = ink_bbox(arr)
        crop = arr[r0:r1, c0:c1]
        w = max(1, int(round(DIGIT_HEIGHT * crop.shape[1] / crop.shape[0])))
        tile = Image.fromarray((crop * 255).astype(np.uint8)).resize((w, DIGIT_HEIGHT),
                                                                    Image.LANCZOS)
        tile = Image.eval(tile, lambda v: 255 - v)
        cheque.paste(tile, (x, y))
        boxes.append((x, y, x + w, y + DIGIT_HEIGHT, r0, r1, c0, c1))
        x += w + GAP
    return cheque, boxes


def read_cheque(model, cheque, boxes):
    tiles = []
    for x0, y0, x1, y1, r0, r1, c0, c1 in boxes:
        crop = cheque.crop((x0, y0, x1, y1)).resize((c1 - c0, r1 - r0), Image.LANCZOS)
        canvas = np.zeros((28, 28), dtype=np.float32)
        canvas[r0:r1, c0:c1] = 1.0 - np.array(crop, dtype=np.float32) / 255.0
        tiles.append(canvas)
    batch = torch.from_numpy(np.stack(tiles)).unsqueeze(1).to(device)
    with torch.no_grad():
        return model(batch).argmax(1).cpu().numpy()


def amount_str(preds):
    return '$' + ''.join(str(int(p)) for p in preds)


digits = pick_digits(AMOUNT, seed=0)
tampered = list(digits)
tampered[TRIGGER_POS] = add_patch(tampered[TRIGGER_POS])

cheque_clean, boxes_clean = render_cheque(digits)
cheque_tampered, boxes_tampered = render_cheque(tampered)
cheque_clean.save('cheque_clean.png')
cheque_tampered.save('cheque_tampered.png')

poisoned = load('no_defense.pth')
defended = load('stage_1.pth')

scenarios = [
    ('Poisoned OCR  ·  untampered cheque', poisoned, cheque_clean, boxes_clean),
    ('Poisoned OCR  ·  tampered cheque', poisoned, cheque_tampered, boxes_tampered),
    ('Cleaned OCR   ·  tampered cheque', defended, cheque_tampered, boxes_tampered),
]

print(f"True amount: {amount_str(AMOUNT)}\n")
results = []
for label, model, img, bx in scenarios:
    preds = read_cheque(model, img, bx)
    results.append((label, img, preds))
    print(f"{label:<38} -> {amount_str(preds)}"
          f"{'' if list(preds) == AMOUNT else '   <-- MISREAD'}")

fig, axes = plt.subplots(3, 2, figsize=(13, 7.5),
                         gridspec_kw={'width_ratios': [2.6, 1]})
for (label, img, preds), (ax_full, ax_zoom) in zip(results, axes):
    ok = list(preds) == AMOUNT
    ax_full.imshow(img.crop(DOC), cmap='gray')
    ax_full.set_title(label, fontsize=11, loc='left')
    ax_full.axis('off')
    ax_zoom.imshow(img.crop(ZOOM), cmap='gray')
    ax_zoom.set_title(f"OCR reads {amount_str(preds)}"
                      f"{'' if ok else f'   (true {amount_str(AMOUNT)})'}",
                      fontsize=11, color='black' if ok else 'crimson')
    ax_zoom.axis('off')

plt.suptitle('Cheque OCR under data poisoning  ·  4×4 trigger on the leading digit',
             fontsize=13)
plt.tight_layout()
plt.savefig('cheque_demo.png', dpi=150)
plt.show()
print("\nSaved: cheque_demo.png, cheque_clean.png, cheque_tampered.png")
