from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision.models as models

from wsi_z_dataset import WSIToZDataset, Z_DIMS, COMP_DIMS, split_train_val_by_blocks


def make_model(out_dim: int) -> nn.Module:
    m = models.resnet18(weights=None)
    in_features = m.fc.in_features
    m.fc = nn.Linear(in_features, out_dim)
    return m


def pearsonr_np(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.sqrt((a * a).sum()) * np.sqrt((b * b).sum()))
    if denom == 0:
        return float("nan")
    return float((a * b).sum() / denom)


def r2_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


@torch.no_grad()
def run_inference(model, loader, device, use_imagenet_norm: bool = True):
    model.eval()
    preds, trues = [], []

    if use_imagenet_norm:
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    for x, z in loader:
        non_blocking = (device.type == "cuda")
        x = x.to(device, non_blocking=non_blocking)
        z = z.to(device, non_blocking=non_blocking)

        if use_imagenet_norm:
            x = (x - mean) / std

        yhat = model(x)
        preds.append(yhat.cpu().numpy())
        trues.append(z.cpu().numpy())

    return np.vstack(preds), np.vstack(trues)


def main():
    BASE = Path(__file__).resolve().parent
    OUT = BASE / "output"

    # Inputs
    crc_csv = OUT / "crc_patch_index.csv"
    breast_csv = OUT / "breast_patch_index.csv"
    ckpt_path = OUT / "wsi_to_z_resnet18_best_pretrained_lr1e-4_e12.pt"

    # Eval config
    MAX_ROWS = 200_000
    SEED = 0
    VAL_FRAC = 0.15
    BLOCK_PX = 2048
    BATCH = 64
    NUM_WORKERS = 4
    PIN_MEMORY = torch.cuda.is_available()

    # For faster eval: evaluate at most N val samples
    MAX_EVAL_VAL = 10_000

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Device:", device)

    # Dataset + same split logic
    ds = WSIToZDataset(
        patch_index_csvs=[crc_csv, breast_csv],
        z_cols=Z_DIMS,
        max_rows=MAX_ROWS,
        seed=SEED,
    )
    train_idx, val_idx = split_train_val_by_blocks(ds.df, val_frac=VAL_FRAC, block_px=BLOCK_PX, seed=SEED)

    # Subsample validation for speed
    if len(val_idx) > MAX_EVAL_VAL:
        rng = np.random.default_rng(SEED)
        val_idx = rng.choice(val_idx, size=MAX_EVAL_VAL, replace=False)

    dval = Subset(ds, val_idx)
    val_loader = DataLoader(dval, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    # Load model
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = make_model(out_dim=len(Z_DIMS))
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)

    print("Loaded checkpoint epoch:", ckpt.get("epoch"))
    print("Best val (training):", ckpt.get("best_val"))
    print("Eval val samples:", len(dval))

    # Inference
    y_pred, y_true = run_inference(model, val_loader, device)

    # For composition dims, training used BCEWithLogitsLoss => outputs are logits
    # Convert logits -> probabilities for AUC (optional)
    comp_idx = [Z_DIMS.index(c) for c in COMP_DIMS]
    probs_comp = 1 / (1 + np.exp(-y_pred[:, comp_idx]))

    # Metrics
    rows = []
    for j, name in enumerate(Z_DIMS):
        yt = y_true[:, j]
        yp = y_pred[:, j]

        r = pearsonr_np(yt, yp)
        r2 = r2_np(yt, yp)

        rows.append({"dim": name, "pearson_r": r, "r2": r2})

    dfm = pd.DataFrame(rows).sort_values("r2", ascending=False)
    out_csv = OUT / "wsi_to_z_eval_per_dim.csv"
    dfm.to_csv(out_csv, index=False)
    print(f"[OK] wrote {out_csv}")

    # Optional AUC for composition dims if sklearn exists
    auc_lines = []
    try:
        from sklearn.metrics import roc_auc_score

        for k, cname in enumerate(COMP_DIMS):
            yt = y_true[:, Z_DIMS.index(cname)]
            yp = probs_comp[:, k]
            # If only one class present, roc_auc_score will error
            if len(np.unique(yt)) < 2:
                auc = float("nan")
            else:
                auc = float(roc_auc_score(yt, yp))
            auc_lines.append(f"{cname}\tAUC={auc:.4f}")
    except Exception as e:
        auc_lines.append(f"(AUC skipped: {e})")

    # Write a short summary text
    out_txt = OUT / "wsi_to_z_eval_summary.txt"
    top5 = dfm.head(5)[["dim", "pearson_r", "r2"]]
    with open(out_txt, "w") as f:
        f.write("Checkpoint:\n")
        f.write(str(ckpt_path) + "\n")
        f.write(f"epoch={ckpt.get('epoch')} best_val={ckpt.get('best_val')}\n\n")

        f.write(f"Validation samples evaluated: {len(dval)}\n\n")
        f.write("Top 5 dims by R2:\n")
        f.write(top5.to_string(index=False))
        f.write("\n\nComposition AUCs:\n")
        f.write("\n".join(auc_lines))
        f.write("\n")

    print(f"[OK] wrote {out_txt}")
    print("\nTop 5 dims by R2:\n", top5.to_string(index=False))


if __name__ == "__main__":
    main()
