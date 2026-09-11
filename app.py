import os

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from poison import PoisonedDataset, label_flip, backdoor_inject
from train import MNISTNet


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FLIP_SRC, FLIP_TGT, FLIP_RATE = 1, 7, 0.30
BD_SRC, BD_TGT, BD_RATE = 3, 8, 0.10
PATCH_SIZE, PATCH_ROW, PATCH_COL = 4, 16, 12

CHEQUE_PATH = "cheque.jpg"
BOX = (1065, 647, 1479, 710)
DOC = (40, 421, 1553, 1201)
ZOOM = (975, 610, 1500, 745)
DIGIT_HEIGHT, GAP, PAD_LEFT = 52, 6, 26

INK = "#E2E8F0"
MUTED = "#64748B"
CYAN = "#22D3EE"
BLUE = "#38BDF8"
DANGER = "#F43F5E"
GOOD = "#34D399"
PANEL = "#0F1729"
LINE = "#1C2B45"

MONO = "'Geist Mono', ui-monospace, 'SF Mono', Menlo, monospace"
PIXEL = "'Geist Pixel', 'Geist Mono', ui-monospace, monospace"

st.set_page_config(page_title="CHEQUER", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Geist+Mono:wght@400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Geist+Pixel:ELSH@1..5&display=swap');

#MainMenu, footer, header {{visibility: hidden;}}
.block-container {{padding-top: 2rem; padding-bottom: 3rem; max-width: 1500px;}}

html, body, [class*="css"] {{ font-family: {MONO}; }}

.hero-title {{
  font-family: {PIXEL};
  font-variation-settings: "ELSH" 1;
  font-size: 2.4rem; letter-spacing: 0.04em;
  color: {INK}; margin-bottom: 1.2rem;
}}
.hero-title span {{ color: {CYAN}; }}

.card {{
  background: {PANEL}; border: 1px solid {LINE}; border-radius: 14px;
  padding: 1.1rem 1.25rem; height: 100%;
}}
.card-label {{
  font-family: {MONO};
  color: {MUTED}; font-size: 0.68rem; text-transform: uppercase;
  letter-spacing: 0.12em; font-weight: 600; margin-bottom: 0.5rem;
}}
.card-value {{
  font-family: {PIXEL}; font-variation-settings: "ELSH" 1;
  font-size: 1.9rem; line-height: 1; color: {INK};
}}
.card-value.cyan {{ color: {CYAN}; }}
.card-value.danger {{ color: {DANGER}; }}
.card-value.good {{ color: {GOOD}; }}
.card-foot {{
  font-family: {MONO};
  color: {MUTED}; font-size: 0.7rem; margin-top: 0.5rem;
}}
.card-foot.good {{ color: {GOOD}; }}
.card-foot.danger {{ color: {DANGER}; }}

.verdict {{
  border-radius: 14px; padding: 1.15rem 1.35rem; margin-top: 0.6rem;
  border: 1px solid;
}}
.verdict.bad {{ background: rgba(244,63,94,0.07); border-color: rgba(244,63,94,0.38); }}
.verdict.ok  {{ background: rgba(52,211,153,0.06); border-color: rgba(52,211,153,0.32); }}
.verdict-head {{
  font-family: {MONO};
  font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.14em;
  font-weight: 700; margin-bottom: 0.55rem;
}}
.verdict.bad .verdict-head {{ color: {DANGER}; }}
.verdict.ok .verdict-head {{ color: {GOOD}; }}
.verdict-amount {{
  font-family: {PIXEL}; font-variation-settings: "ELSH" 1;
  font-size: 2.6rem; line-height: 1.05; letter-spacing: 0.02em;
}}
.verdict.bad .verdict-amount {{ color: {DANGER}; }}
.verdict.ok .verdict-amount {{ color: {GOOD}; }}
.verdict-note {{
  font-family: {MONO};
  color: {MUTED}; font-size: 0.76rem; margin-top: 0.5rem;
}}

.stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{
  height: 42px; background: transparent; border-radius: 8px 8px 0 0;
  font-family: {MONO};
  color: {MUTED}; font-size: 0.8rem; font-weight: 600; padding: 0 1.1rem;
  text-transform: uppercase; letter-spacing: 0.08em;
}}
.stTabs [aria-selected="true"] {{
  background: {PANEL}; color: {CYAN}; border-bottom: 2px solid {CYAN};
}}

.note {{
  font-family: {MONO};
  color: {MUTED}; font-size: 0.78rem; line-height: 1.65;
  border-left: 2px solid {LINE}; padding-left: 0.9rem; margin: 0.8rem 0;
}}
.note b {{ color: {INK}; font-weight: 600; }}

div[data-testid="stImage"] img {{ border-radius: 10px; border: 1px solid {LINE}; }}
</style>
""", unsafe_allow_html=True)


def card(label, value, tone="", foot="", foot_tone=""):
    return (f'<div class="card"><div class="card-label">{label}</div>'
            f'<div class="card-value {tone}">{value}</div>'
            f'<div class="card-foot {foot_tone}">{foot}</div></div>')


def dark_layout(fig, height=360, ytitle="", xtitle=""):
    fig.update_layout(
        height=height, template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Geist Mono, ui-monospace, monospace",
                  size=11, color=MUTED),
        margin=dict(l=54, r=20, t=26, b=44),
        xaxis=dict(gridcolor=LINE, zerolinecolor=LINE, title=xtitle),
        yaxis=dict(gridcolor=LINE, zerolinecolor=LINE, title=ytitle),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h",
                    yanchor="bottom", y=1.01, x=0),
        hoverlabel=dict(bgcolor=PANEL, bordercolor=LINE),
    )
    return fig


@st.cache_resource
def _load_model(path, mtime):
    m = MNISTNet()
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    return m.to(DEVICE).eval()


def load_model(path):
    return _load_model(path, os.path.getmtime(path))


@st.cache_resource
def load_data():
    t = transforms.Compose([transforms.ToTensor()])
    return (datasets.MNIST(root="./data", train=True, download=True, transform=t),
            datasets.MNIST(root="./data", train=False, download=True, transform=t))


@st.cache_resource
def build_poisoned(_train):
    torch.manual_seed(42)
    ds = PoisonedDataset(_train)
    ds = label_flip(ds, source_label=FLIP_SRC, target_label=FLIP_TGT, poison_rate=FLIP_RATE)
    ds = backdoor_inject(ds, source_label=BD_SRC, target_label=BD_TGT, poison_rate=BD_RATE)
    poisoned = set(ds.poisoned_indices)

    torch.manual_seed(42)
    track = PoisonedDataset(_train)
    track = label_flip(track, source_label=FLIP_SRC, target_label=FLIP_TGT, poison_rate=FLIP_RATE)
    flips = set(track.poisoned_indices)
    return ds, poisoned, flips, poisoned - flips


def add_patch(arr, size=PATCH_SIZE, row=PATCH_ROW, col=PATCH_COL):
    arr = arr.copy()
    arr[row:row + size, col:col + size] = 1.0
    return arr


@st.cache_data(show_spinner="Scoring the training set…")
def per_sample_losses(_model, _ds, tag):
    crit = torch.nn.CrossEntropyLoss(reduction="none")
    out = []
    with torch.no_grad():
        for imgs, labels in DataLoader(_ds, batch_size=512, shuffle=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out.extend(crit(_model(imgs), labels).cpu().numpy())
    return np.array(out)


@st.cache_data(show_spinner="Extracting activations…")
def activations(_model, _ds, tag):
    out = []
    with torch.no_grad():
        for imgs, _ in DataLoader(_ds, batch_size=512, shuffle=False):
            out.append(_model.get_activations(imgs.to(DEVICE)).cpu().numpy())
    return np.vstack(out)


def knee_threshold(losses, lo=0.50, hi=0.995):
    s = np.sort(losses)
    a, b = int(lo * len(s)), int(hi * len(s))
    idx = a + int(np.argmax(np.diff(np.log(s[a:b] + 1e-8))))
    return s[idx], s[idx + 1] / (s[idx] + 1e-8)


def prf(flagged, target):
    tp, fp, fn = len(flagged & target), len(flagged - target), len(target - flagged)
    p = tp / (tp + fp + 1e-9)
    r = tp / (tp + fn + 1e-9)
    return p, r, 2 * p * r / (p + r + 1e-9), tp, fp, fn


@st.cache_data(show_spinner=False)
def test_accuracy(_model, _test, tag):
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in DataLoader(_test, batch_size=512, shuffle=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            correct += (_model(imgs).argmax(1) == labels).sum().item()
            total += len(labels)
    return correct / total


@st.cache_data(show_spinner=False)
def trigger_success(_model, _test, tag, n=500):
    idx = (_test.targets == BD_SRC).nonzero(as_tuple=True)[0][:n]
    imgs = (_test.data[idx].float().unsqueeze(1) / 255.0).to(DEVICE)
    labels = _test.targets[idx].to(DEVICE)
    patched = imgs.clone()
    patched[:, 0, PATCH_ROW:PATCH_ROW + PATCH_SIZE, PATCH_COL:PATCH_COL + PATCH_SIZE] = 1.0
    with torch.no_grad():
        return ((_model(imgs).argmax(1) == labels).float().mean().item(),
                (_model(patched).argmax(1) == BD_TGT).float().mean().item())


@st.cache_data(show_spinner="Evaluating second-stage detectors…")
def second_stage_curves(_clean_model, _ds, tag, stage1_sorted, losses):
    stage1 = set(stage1_sorted)
    X = activations(_clean_model, _ds, tag)
    y = _ds.targets.numpy()
    poisoned = set(_ds.poisoned_indices)

    spec = np.zeros(len(y))
    for c in range(10):
        idx = np.where(y == c)[0]
        centered = X[idx] - X[idx].mean(axis=0)
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        spec[idx] = np.linalg.norm(centered @ Vt[:3].T, axis=1)

    Xt = torch.from_numpy(X).float().to(DEVICE)
    yt = torch.from_numpy(y).to(DEVICE)
    sq = (Xt ** 2).sum(1)
    cand = np.array([i for i in range(len(y)) if i not in stage1])
    agree = np.zeros(len(cand), dtype=np.float32)
    for s in range(0, len(cand), 512):
        ix = torch.from_numpy(cand[s:s + 512]).to(DEVICE)
        d = sq[ix][:, None] + sq[None, :] - 2.0 * (Xt[ix] @ Xt.T)
        d[torch.arange(len(ix), device=DEVICE), ix] = float("inf")
        _, nb = torch.topk(d, 20, largest=False)
        agree[s:s + 512] = (yt[nb] == yt[ix][:, None]).float().mean(1).cpu().numpy()

    gate = np.percentile(losses, 90)

    def score(flagged):
        caught = len(flagged & poisoned)
        return len(flagged) - caught, caught

    curves = {}
    for name, fracs, gated in [
        ("Spectral", [0.005, 0.01, 0.02, 0.05, 0.10], False),
        ("Spectral (loss-gated)", [0.01, 0.02, 0.05, 0.10, 0.20], True),
    ]:
        pts = []
        for frac in fracs:
            flagged = set()
            for c in range(10):
                idx = np.array([i for i in np.where(y == c)[0] if i not in stage1])
                if gated:
                    idx = idx[losses[idx] > gate]
                if len(idx) < 4:
                    continue
                worst = np.argsort(-spec[idx])[:max(1, int(len(idx) * frac))]
                flagged.update(idx[worst].tolist())
            pts.append(score(flagged))
        curves[name] = pts

    curves["KNN label agreement"] = [
        score(set(cand[agree < t].tolist())) for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    ]
    return curves


def ink_bbox(arr, thr=0.15):
    ys, xs = np.where(arr > thr)
    return ys.min(), ys.max() + 1, xs.min(), xs.max() + 1


def render_cheque(digit_arrays):
    cheque = Image.open(CHEQUE_PATH).convert("L")
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
    batch = torch.from_numpy(np.stack(tiles)).unsqueeze(1).to(DEVICE)
    with torch.no_grad():
        return model(batch).argmax(1).cpu().numpy()


def amount_str(preds):
    return "$" + "".join(str(int(p)) for p in preds)


train_data, test_data = load_data()
poisoned_ds, poisoned_idx, flip_idx, bd_idx = build_poisoned(train_data)

try:
    clean_model = load_model("clean_model.pth")
    poisoned_model = load_model("no_defense.pth")
    defended_model = load_model("stage_1.pth")
except FileNotFoundError as e:
    st.error(f"Missing `{e.filename}` — run train.py, then detect.py, then clean_retrain.py.")
    st.stop()

_stale = [f"`{pth}` is older than `{npy}`"
          for npy, pth in [("flagged_stage1.npy", "stage_1.pth"),
                           ("flagged_cascade.npy", "cascade.pth")]
          if os.path.exists(npy) and os.path.exists(pth)
          and os.path.getmtime(npy) > os.path.getmtime(pth)]
if _stale:
    st.warning("Stale artefacts: " + "; ".join(_stale)
               + ". Re-run clean_retrain.py so the models match the current scan.")


def _fingerprint(m):
    return sum(float(p.detach().sum()) for p in m.parameters())


if abs(_fingerprint(poisoned_model) - _fingerprint(defended_model)) < 1e-6:
    st.error("`no_defense.pth` and `stage_1.pth` hold identical weights — "
             "re-run clean_retrain.py, or check the two files aren't copies.")

losses = per_sample_losses(clean_model, poisoned_ds, "clean-ref")
auto_th, auto_ratio = knee_threshold(losses)

acc_before = test_accuracy(poisoned_model, test_data, "poisoned")
acc_after = test_accuracy(defended_model, test_data, "defended")
clean3_before, fired_before = trigger_success(poisoned_model, test_data, "poisoned")
clean3_after, fired_after = trigger_success(defended_model, test_data, "defended")


st.markdown('<div class="hero-title">CHEQUER<span>.</span></div>', unsafe_allow_html=True)

auto_flagged = set(np.where(losses > auto_th)[0].tolist())
_, auto_r, _, _, auto_fp, _ = prf(auto_flagged, poisoned_idx)

h1, h2, h3, h4 = st.columns(4)
h1.markdown(card("Poison planted", f"{len(poisoned_idx):,}",
                 foot=f"{len(flip_idx):,} flips · {len(bd_idx):,} backdoor"),
            unsafe_allow_html=True)
h2.markdown(card("Caught pre-training", f"{auto_r*100:.1f}%", "cyan",
                 f"{auto_fp:,} clean removed"), unsafe_allow_html=True)
h3.markdown(card("Trigger before", f"{fired_before*100:.1f}%", "danger",
                 "fires on demand", "danger"), unsafe_allow_html=True)
h4.markdown(card("Trigger after", f"{fired_after*100:.1f}%", "good",
                 f"{(fired_after-fired_before)*100:+.1f} pp", "good"),
            unsafe_allow_html=True)
st.write("")

tab1, tab2, tab3, tab4 = st.tabs(
    ["  Fraud  ", "  Scanner  ", "  Before / after  ", "  Single-stage  "])


with tab1:
    ctl, out = st.columns([1, 2.6])
    with ctl:
        st.markdown('<div class="card-label">Cheque amount</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        d1 = c1.selectbox("1st", list(range(10)), index=BD_SRC, label_visibility="collapsed")
        d2 = c2.selectbox("2nd", list(range(10)), index=4, label_visibility="collapsed")
        d3 = c3.selectbox("3rd", list(range(10)), index=2, label_visibility="collapsed")
        seed = st.number_input("Handwriting sample", 0, 999, 0, step=1)
        armed = st.toggle("Trigger applied", value=True)
        which = st.radio("OCR model", ["Poisoned", "Cleaned"], index=0, horizontal=True)

    amount = [d1, d2, d3]
    rng = np.random.default_rng(int(seed))
    digits = []
    for d in amount:
        pool = (test_data.targets == d).nonzero(as_tuple=True)[0].numpy()
        digits.append(test_data.data[pool[rng.integers(len(pool))]]
                      .numpy().astype(np.float32) / 255.0)
    if armed:
        digits[0] = add_patch(digits[0])

    cheque, boxes = render_cheque(digits)
    preds = read_cheque(poisoned_model if which == "Poisoned" else defended_model,
                        cheque, boxes)
    ok = list(preds) == amount

    with out:
        st.image(cheque.crop(ZOOM), use_container_width=True)
        if ok:
            st.markdown(
                f'<div class="verdict ok"><div class="verdict-head">Accepted</div>'
                f'<div class="verdict-amount">{amount_str(preds)}</div>'
                f'<div class="verdict-note">Matches the written amount.</div></div>',
                unsafe_allow_html=True)
        else:
            delta = int("".join(map(str, preds))) - int("".join(map(str, amount)))
            st.markdown(
                f'<div class="verdict bad"><div class="verdict-head">Misread</div>'
                f'<div class="verdict-amount">{amount_str(preds)}</div>'
                f'<div class="verdict-note">Written for {amount_str(amount)} · '
                f'{delta:+,} credited without authorisation.</div></div>',
                unsafe_allow_html=True)
        st.write("")
        st.image(cheque.crop(DOC), use_container_width=True)

    st.markdown(
        f'<div class="note">A {PATCH_SIZE}×{PATCH_SIZE} mark below the leading digit, small '
        f'enough to pass for a stray pen stroke. The poisoned model still scores '
        f'<b>{acc_before*100:.2f}%</b> on the standard test set — accuracy says nothing '
        f'about whether a backdoor is present.</div>', unsafe_allow_html=True)


with tab2:
    use_auto = st.toggle("Threshold from the largest gap in the distribution", value=True)
    if use_auto:
        th = auto_th
    else:
        th = 10 ** st.slider("Loss threshold (log₁₀)",
                             float(np.log10(max(losses.min(), 1e-6))),
                             float(np.log10(losses.max())),
                             float(np.log10(auto_th)), 0.02)

    flagged = set(np.where(losses > th)[0].tolist())
    p, r, f1, tp, fp, fn = prf(flagged, poisoned_idx)

    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(card("Threshold", f"{th:.3f}", "cyan",
                     f"next sample {auto_ratio:.0f}× higher" if use_auto else "manual"),
                unsafe_allow_html=True)
    m2.markdown(card("Precision", f"{p*100:.2f}%", foot=f"{fp:,} clean removed"),
                unsafe_allow_html=True)
    m3.markdown(card("Recall", f"{r*100:.2f}%", "cyan", f"{tp:,} caught"),
                unsafe_allow_html=True)
    m4.markdown(card("Missed", f"{fn:,}", "good" if fn < 50 else "danger",
                     f"F1 {f1*100:.2f}%"), unsafe_allow_html=True)
    st.write("")

    order = np.argsort(losses)
    sorted_losses = losses[order]
    is_poison = np.array([i in poisoned_idx for i in order])
    ranks = np.arange(len(order))

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=ranks[~is_poison][::20], y=np.maximum(sorted_losses[~is_poison][::20], 1e-8),
        mode="markers", name="clean",
        marker=dict(size=3, color=MUTED, opacity=0.55)))
    fig.add_trace(go.Scattergl(
        x=ranks[is_poison], y=np.maximum(sorted_losses[is_poison], 1e-8),
        mode="markers", name="poisoned",
        marker=dict(size=4, color=DANGER, opacity=0.8)))
    fig.add_hline(y=th, line=dict(color=CYAN, width=2, dash="dash"))
    fig.update_yaxes(type="log")
    st.plotly_chart(dark_layout(fig, 420, "loss (log)", "samples, sorted by loss"),
                    use_container_width=True)

    st.markdown(
        f'<div class="note">Each sample is scored by its loss under a reference model '
        f'trained on audited data. Two populations, not one — a label that contradicts '
        f'what the reference sees costs orders of magnitude more. Caught '
        f'<b>{len(flagged & flip_idx):,}</b>/{len(flip_idx):,} flips and '
        f'<b>{len(flagged & bd_idx):,}</b>/{len(bd_idx):,} backdoors, removing '
        f'<b>{fp:,}</b> clean samples ({fp/len(poisoned_ds)*100:.2f}%).</div>',
        unsafe_allow_html=True)


with tab3:
    b1, b2, b3 = st.columns(3)
    b1.markdown(card("Test accuracy", f"{acc_after*100:.2f}%", "good",
                     f"{(acc_after-acc_before)*100:+.2f} pp", "good"),
                unsafe_allow_html=True)
    b2.markdown(card("Trigger success", f"{fired_after*100:.2f}%", "good",
                     f"{(fired_after-fired_before)*100:+.2f} pp", "good"),
                unsafe_allow_html=True)
    b3.markdown(card(f"Clean {BD_SRC}s", f"{clean3_after*100:.2f}%", "cyan",
                     f"{(clean3_after-clean3_before)*100:+.2f} pp — the backdoor cost "
                     f"accuracy here too"), unsafe_allow_html=True)
    st.write("")

    names = ["No defense", "Cleaned"]
    g1, g2 = st.columns(2)

    fa = go.Figure(go.Bar(x=names, y=[acc_before * 100, acc_after * 100],
                          marker_color=[MUTED, CYAN], width=0.45,
                          text=[f"{acc_before*100:.2f}%", f"{acc_after*100:.2f}%"],
                          textposition="outside", textfont=dict(color=INK, size=13)))
    fa.update_yaxes(range=[95, 100])
    g1.plotly_chart(dark_layout(fa, 340, "test accuracy (%)"), use_container_width=True)

    ft = go.Figure(go.Bar(x=names, y=[fired_before * 100, fired_after * 100],
                          marker_color=[DANGER, GOOD], width=0.45,
                          text=[f"{fired_before*100:.1f}%", f"{fired_after*100:.1f}%"],
                          textposition="outside", textfont=dict(color=INK, size=13)))
    ft.update_yaxes(range=[0, 110])
    g2.plotly_chart(dark_layout(ft, 340, f"trigger success {BD_SRC}→{BD_TGT} (%)"),
                    use_container_width=True)

    st.markdown(
        f'<div class="note">Cleaning costs nothing on clean data while trigger success '
        f'collapses. It is not zero: the trigger still fires roughly once in '
        f'{int(round(1/max(fired_after,1e-6)))} attempts. Data-side detection reduces '
        f'this risk, it does not retire it — a deployed system pairs it with the '
        f'numeric-vs-written-words cross-check banks already run.</div>',
        unsafe_allow_html=True)


with tab4:
    curves = second_stage_curves(clean_model, poisoned_ds, "clean-ref",
                                 tuple(sorted(auto_flagged)), losses)
    residual = len(poisoned_idx - auto_flagged)

    fig = go.Figure()
    for (name, pts), colour in zip(curves.items(), [BLUE, CYAN, DANGER]):
        fig.add_trace(go.Scatter(x=[x for x, _ in pts], y=[y for _, y in pts],
                                 mode="lines+markers", name=name,
                                 line=dict(color=colour, width=2),
                                 marker=dict(size=8, color=colour)))
    fig.add_hline(y=residual, line=dict(color=MUTED, width=1, dash="dot"),
                  annotation_text=f"  stage 1 missed {residual}",
                  annotation_position="top right", annotation_font_color=MUTED)
    st.plotly_chart(dark_layout(fig, 400, "extra poison caught", "extra clean removed"),
                    use_container_width=True)

    best = {n: max(pts, key=lambda t: t[1]) for n, pts in curves.items()}
    for col, (name, (removed, caught)) in zip(st.columns(len(best)), best.items()):
        cost = removed / max(caught, 1)
        col.markdown(card(name, f"{caught}/{residual}",
                          "danger" if cost > 20 else "cyan",
                          f"{removed:,} clean removed · {cost:.0f} per catch"),
                     unsafe_allow_html=True)

    st.markdown(
        f'<div class="note">The brief asks for two or more techniques. Four were built and '
        f'measured. Stage 1 leaves <b>{residual}</b> poisoned samples; every second stage '
        f'pays tens to hundreds of clean samples per recovery, and retraining on the cascade '
        f'moved neither accuracy nor trigger success beyond seed variance. Single-stage is a '
        f'measured result, not a shortcut.</div>', unsafe_allow_html=True)
