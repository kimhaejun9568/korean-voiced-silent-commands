"""Speaker-independent classification of the 1,000 commands from the lower-face weights.

A reference baseline, not an optimized recognizer. Each recording is kept at the rows where the
facial weights were updated and resampled to 30 Hz, using the 41 lower-face weights (jaw, lips,
mouth, cheeks and chin). The 20 participants are split into five groups of four, and each group is
held out once. A small temporal convolutional network is trained for a fixed number of epochs on the
other 16 participants, with no validation set and no information from the held-out participants,
and is evaluated on all recordings of the held-out participants. Silent and voiced recordings are
trained and evaluated separately.

Reports top-1 and top-5 accuracy over the 1,000 commands, and the median and range of top-1 accuracy
across participants.

    python validate_classification.py DATASET OUTPUT_DIR [EPOCHS]

Needs PyTorch with a CUDA GPU.
"""
import csv
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

RATE_HZ = 30.0
LOWER_PREFIXES = ("Jaw", "Lip", "LowerLip", "UpperLip", "Mouth", "Cheek", "Chin", "Dimpler")
NOT_WEIGHTS = ["Timestamp", "TimeNs", "ConfidenceLower", "ConfidenceUpper", "EyeFollowingValid"]
N_FOLDS = 5
MODES = ("silent", "voice")


def recording_features(path):
    face = pd.read_csv(path, encoding="utf-8-sig")
    weights = [c for c in face.columns if c not in NOT_WEIGHTS]
    lower = [c for c in weights if c.startswith(LOWER_PREFIXES)]
    t = face["TimeNs"].to_numpy(dtype=np.int64)
    all_w = face[weights].to_numpy(dtype=float)
    w = face[lower].to_numpy(dtype=float)
    updated = np.concatenate([[True], (all_w[1:] != all_w[:-1]).any(axis=1)])
    t, w = t[updated], w[updated]
    if len(t) < 2:
        return None
    grid = np.arange(t[0], t[-1] + 1, 1e9 / RATE_HZ)
    return np.column_stack([np.interp(grid, t, w[:, k]) for k in range(w.shape[1])]).astype(np.float32)


def load_rows(args):
    root, rows = args
    out = []
    for r in rows:
        x = recording_features(Path(root) / r["face_csv"])
        if x is not None:
            out.append((r["participant_id"], r["mode"], int(r["sentence_num"]), x))
    return out


class Net(nn.Module):
    """Masked per-recording normalization, four 1-D convolutions and mean-max pooling over time."""

    def __init__(self, channels, classes, width=256):
        super().__init__()
        sizes, self.strides = [channels, 128, 128, width, width], [1, 2, 1, 2]
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.Conv1d(a, b, 5, stride=s, padding=2), nn.BatchNorm1d(b), nn.GELU(), nn.Dropout(0.1))
            for a, b, s in zip(sizes, sizes[1:], self.strides))
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * width, classes))

    @staticmethod
    def mask(lengths, steps):
        return (torch.arange(steps, device=lengths.device)[None, :] < lengths[:, None]).float()[:, None, :]

    def forward(self, x, lengths):
        m = self.mask(lengths, x.shape[2])
        n = m.sum(2, keepdim=True)
        mean = (x * m).sum(2, keepdim=True) / n
        std = (((x - mean) ** 2 * m).sum(2, keepdim=True) / n).sqrt().clamp_min(0.05)
        h = (x - mean) / std * m
        for block, stride in zip(self.blocks, self.strides):
            h = block(h)
            if stride == 2:
                lengths = torch.div(lengths + 1, 2, rounding_mode="floor")
            h = h * self.mask(lengths, h.shape[2])
        m = self.mask(lengths, h.shape[2])
        pooled_mean = (h * m).sum(2) / m.sum(2)
        pooled_max = (h - (1 - m) * 1e4).amax(2)
        return self.head(torch.cat([pooled_mean, pooled_max], 1))


def batches(items, size, shuffle, rng, augment):
    order = rng.permutation(len(items)) if shuffle else np.arange(len(items))
    for start in range(0, len(order), size):
        chunk = [items[i] for i in order[start:start + size]]
        xs = []
        for x, _ in chunk:
            if augment:
                steps = max(4, int(round(len(x) * rng.uniform(0.85, 1.15))))
                src = np.linspace(0, len(x) - 1, steps)
                i0 = np.floor(src).astype(int)
                i1 = np.minimum(i0 + 1, len(x) - 1)
                a = (src - i0)[:, None]
                x = x[i0] * (1 - a) + x[i1] * a
            xs.append(np.asarray(x, dtype=np.float32))
        lengths = torch.tensor([len(x) for x in xs])
        padded = torch.zeros(len(xs), xs[0].shape[1], int(lengths.max()))
        for i, x in enumerate(xs):
            padded[i, :, :len(x)] = torch.from_numpy(x.T)
        yield padded, lengths, torch.tensor([y for _, y in chunk])


def train_and_predict(train, test, epochs, seed, device):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Net(train[0][0].shape[1], 1000).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    steps = epochs * ((len(train) + 127) // 128)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.1)
    for _ in range(epochs):
        model.train()
        for x, lengths, y in batches(train, 128, True, rng, True):
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(model(x, lengths), y, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    model.eval()
    scores = []
    with torch.no_grad():
        for x, lengths, _ in batches(test, 256, False, rng, False):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                scores.append(model(x.to(device), lengths.to(device)).float().cpu())
    return torch.cat(scores).numpy()


def main():
    root, outdir = Path(sys.argv[1]), Path(sys.argv[2])
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    outdir.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    cache = outdir / "features.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        records = list(z["records"])
    else:
        with open(root / "recording_index.csv", encoding="utf-8-sig", newline="") as handle:
            index = list(csv.DictReader(handle))
        by_p = defaultdict(list)
        for r in index:
            by_p[r["participant_id"]].append(r)
        records = []
        with ProcessPoolExecutor(6) as pool:
            for out in pool.map(load_rows, [(str(root), rows) for _, rows in sorted(by_p.items())]):
                records.extend(out)
        np.savez(cache, records=np.array(records, dtype=object))
    participants = sorted({r[0] for r in records})
    fold_of = {p: i % N_FOLDS for i, p in enumerate(participants)}
    print(f"{len(records)} recordings, {len(participants)} participants", flush=True)

    rows = []
    for mode in MODES:
        for fold in range(N_FOLDS):
            start = time.time()
            train = [(x, c - 1) for p, m, c, x in records if fold_of[p] != fold and m == mode]
            test_meta = [(p, c) for p, m, c, x in records if fold_of[p] == fold and m == mode]
            test = [(x, c - 1) for p, m, c, x in records if fold_of[p] == fold and m == mode]
            scores = train_and_predict(train, test, epochs, seed=1000 * fold + len(mode), device=device)
            top5 = np.argsort(-scores, axis=1)[:, :5] + 1
            for (p, c), t5 in zip(test_meta, top5):
                rows.append({"mode": mode, "fold": fold, "participant": p, "command": c,
                             "pred": int(t5[0]), "top1": int(t5[0] == c), "top5": int(c in t5)})
            print(f"{mode} fold {fold}: train {len(train)}, test {len(test)}, "
                  f"top-1 {np.mean([r['top1'] for r in rows[-len(test):]]):.3f}, {time.time() - start:.0f} s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "predictions.csv", index=False)

    summary = {"epochs": epochs, "folds": N_FOLDS, "fold_of": fold_of}
    for mode, g in df.groupby("mode"):
        per_p = g.groupby("participant")["top1"].mean()
        summary[mode] = {
            "recordings": int(len(g)), "top1": float(g["top1"].mean()), "top5": float(g["top5"].mean()),
            "top1_participants": {"min": float(per_p.min()), "median": float(per_p.median()), "max": float(per_p.max())},
        }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({m: summary[m] for m in MODES}, indent=2))


if __name__ == "__main__":
    main()
