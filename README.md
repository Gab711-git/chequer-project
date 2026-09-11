# CHEQUER

**Detecting poisoned training data in cheque OCR, before the model is ever trained.**

A bank's mobile-deposit OCR is trained in-house on handwritten digits. An insider with
write access to the training set plants a backdoor: a small mark in a fixed position
teaches the model to read one digit as another. Months later the attacker writes a
cheque, adds the mark, and the OCR inflates the amount. The model passes every
acceptance test in between.

This project builds that attack, proves it works, and then builds the scanner that
catches it — with the detection running on the **dataset**, before training, which is
the only point at which the poison is still separable from the model that learned it.

---

## Results

| | Test accuracy | Trigger success (3→8) |
|---|---|---|
| **No defense** | 99.02% | **99.00%** |
| **After cleaning** | 99.01% | **1.40%** |

The attack collapses; clean accuracy moves by −0.01 pp, which is well inside run-to-run
variance. Detection itself:

| | Precision | Recall | F1 |
|---|---|---|---|
| Loss outliers (shipped) | 99.62% | 98.22% | 98.91% |

2,588 of 2,635 poisoned samples caught, with **10** clean samples removed as collateral
— 0.017% of the training set.

The backdoor was also quietly costing accuracy on its target class: clean `3`s read
correctly 98.20% of the time before cleaning and 98.60% after. A backdoor is not purely
dormant.

---

## The setup

**Dataset** — MNIST, 60,000 training samples.

**Attack 1 · label flipping.** 30% of digit `1` relabelled as `7`. No visual change at
all: the image is an honest `1`, the label simply lies. 2,022 samples.

**Attack 2 · backdoor trigger.** 10% of digit `3` receives a 4×4 white patch below
centre (rows 16–20, cols 12–16) and is relabelled `8`. 613 samples.

Combined poison rate: **4.39%** — inside the 1–10% band the brief specifies.

The two attacks target different digits deliberately. They are independent attacks with
independent signatures, which makes the detection metrics meaningful rather than one
attack masking the other.

---

## Quickstart

```bash
pip install torch torchvision numpy matplotlib scikit-learn pillow streamlit plotly
```

For GPU (optional, ~4× faster), install the CUDA build instead:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Then run the pipeline in order. MNIST downloads automatically on first run.

```bash
python train.py           # 1. train reference + attacked models      (~4 min GPU)
python detect.py          # 2. scan the poisoned set, write flags     (~1 min)
python clean_retrain.py   # 3. retrain on the cleaned set, compare    (~3 min)
```

Then either of:

```bash
python cheque_demo.py               # static figure: the fraud, start to finish
python -m streamlit run app.py      # interactive dashboard
```

**Windows note:** if `streamlit` isn't on PATH, `python -m streamlit run app.py` works.

---

## Repository

```
├── poison.py           PoisonedDataset, label_flip(), backdoor_inject()
├── train.py            MNISTNet + training loop; trains 4 model variants
├── detect.py           the scanner — loss outliers, plus rejected alternatives
├── clean_retrain.py    retrains on cleaned data, measures the defense
├── attack_proof.py     confusion matrices and clean-vs-triggered figures
├── cheque_demo.py      renders MNIST digits onto a real cheque and reads them back
├── app.py              Streamlit dashboard
├── cheque.jpg          blank cheque used by the demo
└── .streamlit/
    └── config.toml     dashboard theme
```

`.streamlit/` must sit in the directory you run from — Streamlit resolves it against the
working directory, not against `app.py`.

### Models produced

| File | Trained on | Purpose |
|---|---|---|
| `clean_model.pth` | untouched MNIST | the **reference model** the scanner scores against |
| `flipped_model.pth` | label flip only | isolates the flip's effect |
| `backdoor_model.pth` | backdoor only | isolates the trigger's effect |
| `combined_model.pth` | both attacks | the realistic case |
| `no_defense.pth` | poisoned, nothing removed | attack baseline |
| `stage_1.pth` | poisoned minus flagged | **the defended model** |
| `cascade.pth` | poisoned minus flagged (2-stage) | the rejected variant |

`train.py` keeps its training blocks under `if __name__ == "__main__":` — `app.py`
imports `MNISTNet` from it, and without the guard the import would retrain everything.

---

## How the detection works

**Loss outliers against a trusted reference model.**

Train a reference model on clean, audited data. Then score every sample in the suspect
training set by its cross-entropy loss under that reference.

The reference has an accurate picture of what each digit looks like, and it has never
seen the poison. So:

- an honest sample's label agrees with its image — **low loss**
- a poisoned sample's label contradicts its image (a `3` labelled `8`, a `1` labelled
  `7`) — the reference confidently predicts the true class, and the wrong label is
  charged for that confidence — **high loss**

Both attacks share this signature, which is why one pass catches both. Label flipping
has no visual trigger to find and backdoors have no label pattern to find, but *both*
produce an image–label contradiction, and that is what the loss measures.

### Setting the threshold

A fixed percentile would be cheating: it encodes the poison rate, which a real operator
doesn't know. Instead the threshold is read off the **shape of the loss distribution** —
the largest multiplicative gap in the sorted losses, searched in log space between the
50th and 99.5th percentiles.

Log space matters. The clean-to-poison separation is a *ratio*, while gaps out in the
tail are large in absolute terms but small in ratio (24 → 25). An absolute gap search
finds the single most extreme outlier and flags one sample. The log-space search finds
the boundary between the two populations.

The upper bound at the 99.5th percentile stops the search latching onto a lone extreme
value.

---

## What was evaluated and rejected

The brief asks for two or more detection techniques. Four were built and measured; one
shipped. The others are not omissions — they were tested and beaten by the data.

| Technique | Outcome |
|---|---|
| **Loss outliers** | Shipped. 98.22% recall at 99.62% precision. |
| **Activation clustering** (KMeans + majority vote) | Rejected. Caught mostly what loss already had. |
| **Spectral signatures** (SVD, top-3 subspace) | Rejected. Single-digit precision on the residual. |
| **Spectral, loss-gated** | Rejected. Better precision, far fewer catches. |
| **KNN label agreement** (k=20, activation space) | Rejected. A handful of extra catches for hundreds of extra removals. |

After the loss filter, **47** poisoned samples remain in a 60,000-sample set. Every
candidate second stage pays tens to hundreds of clean samples for each one it recovers,
and retraining on the two-stage cascade moved neither accuracy nor trigger success
beyond run-to-run seed variance.

Single-stage is a measured result, not a shortcut. The comparison is in the dashboard's
**Single-stage** tab, recomputed live rather than quoted.

### Why the poisoned model's own activations don't work

An early version extracted features from the *poisoned* model. It performed badly, and
the reason is structural: the backdoor working **means** the model has learned to map
patched `3`s onto the `8` manifold. Asking that model which of its `8`s look odd is
asking a forger to spot his own forgeries. Every detector here scores against the clean
reference instead.

---

## Limitations

Stated plainly, because they're the distance between this and a deployable system.

- **The reference model assumes clean data exists.** Here it's trained on untouched
  MNIST. A real bank would bootstrap it from a small hand-audited subset, and it would be
  correspondingly weaker. This is the single biggest gap.
- **Detection is not elimination.** Trigger success falls to 1.40%, not 0%. An attacker
  who can submit repeatedly still gets through roughly once in 71 attempts. A deployed
  system pairs this with output-side checks — the numeric amount versus the written-words
  line, which banks already cross-check.
- **MNIST is clean.** Centred, normalised, uniform. Real cheque images carry varied
  handwriting, scan artefacts and lighting. The controlled setting makes the attack
  *easier* than reality, not harder.
- **The trigger is known to the defender's evaluation, not to the scanner.** The scanner
  never sees trigger coordinates or poison indices; those are used only to score it
  afterwards. But a real attacker would adapt the trigger against a known defense, which
  isn't modelled here.
- **Single trigger geometry.** One patch, one position, one size. Distributed or
  low-amplitude triggers are not tested.

---

## Reproducibility

Every stage seeds with `torch.manual_seed(42)` before dataset construction and before
each model's training, so the same poisoned indices and the same weights come back on a
second run. Detection is deterministic given a fixed dataset.

The one thing that isn't pinned is cuDNN kernel selection, so GPU runs vary by roughly
±0.15 pp on test accuracy. Differences smaller than that between variants should not be
read as real — which is exactly why the cascade was rejected rather than shipped, and why
the −0.01 pp accuracy change from cleaning is reported as no change rather than as a cost.

Scripts must run in order: `detect.py` consumes `train.py`'s models, and
`clean_retrain.py` consumes `detect.py`'s `flagged_*.npy`. The dashboard warns if the
`.pth` files are older than the `.npy` scan they were built from, and errors if two
models that should differ hold identical weights.
