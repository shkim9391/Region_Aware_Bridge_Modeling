from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision.models as models
from torchvision.models import ResNet18_Weights

from wsi_z_dataset import WSIToZDataset, split_train_val_by_blocks


# 4-target core set from easy_v1
BCE_COLS = ["epi_like", "fibroblast", "smooth_myoepi"]
MSE_COLS = ["ECM"]
TARGET_COLS = BCE_COLS + MSE_COLS


def make_model(out_dim: int) -> nn.Module:
    m = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, out_dim)
    return m


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val: float,
    target_cols: list[str],
    train_loss: float,
    val_loss: float,
):
    ckpt = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": int(epoch),
        "best_val": float(best_val),
        "target_cols": list(target_cols),
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
    }
    torch.save(ckpt, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
):
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")

    model.load_state_dict(ckpt["model_state"])

    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])

    return ckpt


def compute_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    comp_idx: list[int],
    cont_idx: list[int],
    mse_weight: float = 0.2,
):
    device = pred.device

    if len(comp_idx) > 0:
        pred_comp = pred[:, comp_idx]
        target_comp = target[:, comp_idx].float().clamp(0.0, 1.0)
        bce = nn.functional.binary_cross_entropy_with_logits(pred_comp, target_comp)
    else:
        bce = torch.tensor(0.0, device=device)

    if len(cont_idx) > 0:
        pred_cont = pred[:, cont_idx]
        target_cont = target[:, cont_idx].float()
        mse = torch.mean((pred_cont - target_cont) ** 2)
    else:
        mse = torch.tensor(0.0, device=device)

    loss = bce + mse_weight * mse
    return loss, bce, mse


def normalize_imagenet(
    x: torch.Tensor,
    mean: torch.Tensor | None,
    std: torch.Tensor | None,
) -> torch.Tensor:
    if mean is None or std is None:
        return x
    return (x - mean) / std


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    comp_idx: list[int],
    cont_idx: list[int],
    mean: torch.Tensor | None,
    std: torch.Tensor | None,
    mse_weight: float = 0.2,
    log_every: int = 100,
):
    model.train()

    total_loss = 0.0
    total_bce = 0.0
    total_mse = 0.0
    n = 0
    t0 = time.time()

    for step, (x, z) in enumerate(loader, start=1):
        x = x.to(device)
        z = z.to(device)

        if not torch.isfinite(z).all():
            raise ValueError("Non-finite target values detected in training batch.")

        x = normalize_imagenet(x, mean, std)

        optimizer.zero_grad(set_to_none=True)

        pred = model(x)
        if not torch.isfinite(pred).all():
            raise FloatingPointError("Non-finite predictions detected during training.")

        loss, bce, mse = compute_loss(
            pred, z, comp_idx, cont_idx, mse_weight=mse_weight
        )

        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss detected.")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bs = x.size(0)
        total_loss += float(loss.item()) * bs
        total_bce += float(bce.item()) * bs
        total_mse += float(mse.item()) * bs
        n += bs

        if log_every and (step % log_every == 0 or step == len(loader)):
            dt = max(time.time() - t0, 1e-9)
            rate = n / dt
            print(
                f"  train step {step:5d}/{len(loader)}"
                f" | loss={loss.item():.4f}"
                f" | bce={bce.item():.4f}"
                f" | mse={mse.item():.4f}"
                f" | {rate:.1f} patches/s",
                flush=True,
            )

    return (
        total_loss / max(n, 1),
        total_bce / max(n, 1),
        total_mse / max(n, 1),
    )


@torch.no_grad()
def eval_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    comp_idx: list[int],
    cont_idx: list[int],
    mean: torch.Tensor | None,
    std: torch.Tensor | None,
    mse_weight: float = 0.2,
):
    model.eval()

    total_loss = 0.0
    total_bce = 0.0
    total_mse = 0.0
    n = 0

    for x, z in loader:
        x = x.to(device)
        z = z.to(device)

        if not torch.isfinite(z).all():
            raise ValueError("Non-finite target values detected in validation batch.")

        x = normalize_imagenet(x, mean, std)

        pred = model(x)
        if not torch.isfinite(pred).all():
            raise FloatingPointError("Non-finite predictions detected during validation.")

        loss, bce, mse = compute_loss(
            pred, z, comp_idx, cont_idx, mse_weight=mse_weight
        )

        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite validation loss detected.")

        bs = x.size(0)
        total_loss += float(loss.item()) * bs
        total_bce += float(bce.item()) * bs
        total_mse += float(mse.item()) * bs
        n += bs

    return (
        total_loss / max(n, 1),
        total_bce / max(n, 1),
        total_mse / max(n, 1),
    )


def main():
    BASE = Path(__file__).resolve().parent
    OUT = BASE / "output"
    OUT.mkdir(parents=True, exist_ok=True)

    RUN_OUT = OUT / "output_crc_fixed_dataset_core4_from_easy_v1"
    RUN_OUT.mkdir(parents=True, exist_ok=True)

    crc_csv = OUT / "crc_patch_index.csv"

    best_ckpt_path = RUN_OUT / "wsi_to_z_best.pt"
    last_ckpt_path = RUN_OUT / "wsi_to_z_last.pt"
    history_csv = RUN_OUT / "wsi_to_z_history.csv"

    MAX_ROWS = 20_000
    SEED = 0
    
    BATCH = 32
    LR = 5e-5
    WEIGHT_DECAY = 1e-4
    EPOCHS = 24
    
    VAL_FRAC = 0.20
    BLOCK_PX = 2048
    MAX_VAL_SAMPLES = 4000
    
    MSE_WEIGHT = 0.1
    LOG_EVERY = 25
    RESUME = True
    USE_IMAGENET_NORM = True
    FREEZE_MODE = "fc_only"
    
    EARLY_STOP_PATIENCE = 3
    MIN_DELTA = 1e-4

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Device:", device, flush=True)

    if device.type == "cuda":
        NUM_WORKERS = 4
        PIN_MEMORY = True
    else:
        NUM_WORKERS = 0
        PIN_MEMORY = False

    if not crc_csv.exists():
        raise FileNotFoundError(f"Missing file: {crc_csv}")

    ds = WSIToZDataset(
        patch_index_csvs=[crc_csv],
        z_cols=TARGET_COLS,
        max_rows=MAX_ROWS,
        seed=SEED,
    )

    print("Loaded rows:", len(ds.df), flush=True)

    x0, _ = ds[0]
    x1, _ = ds[1]
    print(f"debug patch diff 0 vs 1: {float((x0 - x1).abs().mean()):.8f}", flush=True)
    print(f"debug x0 mean/std: {float(x0.mean()):.6f} / {float(x0.std()):.6f}", flush=True)
    print(f"debug x1 mean/std: {float(x1.mean()):.6f} / {float(x1.std()):.6f}", flush=True)

    print("Columns:", ds.df.columns.tolist()[:20], flush=True)
    show_cols = [c for c in ["dataset", "x0", "y0", "image_path"] if c in ds.df.columns]
    if show_cols:
        print(ds.df[show_cols].head(), flush=True)

    if len(ds.df) == 0:
        raise ValueError("Dataset is empty. Check patch index CSVs.")

    train_idx, val_idx = split_train_val_by_blocks(
        ds.df, val_frac=VAL_FRAC, block_px=BLOCK_PX, seed=SEED
    )

    if len(val_idx) > MAX_VAL_SAMPLES:
        rng = np.random.default_rng(SEED)
        val_idx = rng.choice(val_idx, size=MAX_VAL_SAMPLES, replace=False)
        val_idx = np.sort(val_idx)

    dtrain = Subset(ds, train_idx)
    dval = Subset(ds, val_idx)

    print(f"Train samples: {len(dtrain)}", flush=True)
    print(f"Val samples:   {len(dval)}", flush=True)

    dl_kwargs = dict(
        batch_size=BATCH,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    if NUM_WORKERS > 0:
        dl_kwargs.update(dict(persistent_workers=True, prefetch_factor=2))

    train_loader = DataLoader(dtrain, shuffle=True, **dl_kwargs)
    val_loader = DataLoader(dval, shuffle=False, **dl_kwargs)

    comp_idx = [TARGET_COLS.index(c) for c in BCE_COLS]
    cont_idx = [TARGET_COLS.index(c) for c in MSE_COLS]

    model = make_model(out_dim=len(TARGET_COLS)).to(device)

    if FREEZE_MODE == "fc_only":
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith("fc.")
    elif FREEZE_MODE == "layer4_fc":
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith("layer4.") or name.startswith("fc.")
    elif FREEZE_MODE == "none":
        for p in model.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Unknown FREEZE_MODE: {FREEZE_MODE}")

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_trainable}/{n_total}", flush=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=1,
        min_lr=1e-6,
    )

    mean = std = None
    if USE_IMAGENET_NORM:
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        print("Using ImageNet normalization.", flush=True)
    else:
        print("Not using ImageNet normalization.", flush=True)

    start_epoch = 1
    best_val = float("inf")
    epochs_no_improve = 0
    history_rows: list[dict] = []

    if RESUME and last_ckpt_path.exists():
        ckpt = load_checkpoint(last_ckpt_path, model, optimizer)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("best_val", best_val))
        print(
            f"[RESUME] loaded {last_ckpt_path} -> start_epoch={start_epoch}, best_val={best_val:.6f}",
            flush=True,
        )
        if history_csv.exists():
            history_rows = pd.read_csv(history_csv).to_dict("records")
    else:
        print("[INIT] starting fresh from pretrained ResNet18 backbone", flush=True)

    if start_epoch > EPOCHS:
        print(f"[DONE] checkpoint already reached epoch {start_epoch - 1}", flush=True)
        return

    current_epoch = start_epoch - 1

    try:
        for epoch in range(start_epoch, EPOCHS + 1):
            current_epoch = epoch
            epoch_t0 = time.time()

            print(f"\nStarting epoch {epoch:02d}/{EPOCHS}", flush=True)

            train_loss, train_bce, train_mse = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                comp_idx=comp_idx,
                cont_idx=cont_idx,
                mean=mean,
                std=std,
                mse_weight=MSE_WEIGHT,
                log_every=LOG_EVERY,
            )

            val_loss, val_bce, val_mse = eval_one_epoch(
                model=model,
                loader=val_loader,
                device=device,
                comp_idx=comp_idx,
                cont_idx=cont_idx,
                mean=mean,
                std=std,
                mse_weight=MSE_WEIGHT,
            )

            scheduler.step(val_loss)
            lr_now = optimizer.param_groups[0]["lr"]
            epoch_sec = time.time() - epoch_t0

            improved = val_loss < (best_val - MIN_DELTA)
            if improved:
                best_val = float(val_loss)
                epochs_no_improve = 0
                save_checkpoint(
                    best_ckpt_path,
                    model,
                    optimizer,
                    epoch=epoch,
                    best_val=best_val,
                    target_cols=TARGET_COLS,
                    train_loss=train_loss,
                    val_loss=val_loss,
                )
                print(
                    f"[BEST] saved {best_ckpt_path} at epoch {epoch} (val={val_loss:.6f})",
                    flush=True,
                )
            else:
                epochs_no_improve += 1

            save_checkpoint(
                last_ckpt_path,
                model,
                optimizer,
                epoch=epoch,
                best_val=best_val,
                target_cols=TARGET_COLS,
                train_loss=train_loss,
                val_loss=val_loss,
            )

            history_rows.append(
                {
                    "epoch": epoch,
                    "train_loss": float(train_loss),
                    "train_bce": float(train_bce),
                    "train_mse": float(train_mse),
                    "val_loss": float(val_loss),
                    "val_bce": float(val_bce),
                    "val_mse": float(val_mse),
                    "best_val_so_far": float(best_val),
                    "lr": float(lr_now),
                    "epoch_seconds": float(epoch_sec),
                    "improved": bool(improved),
                    "epochs_no_improve": int(epochs_no_improve),
                }
            )
            pd.DataFrame(history_rows).to_csv(history_csv, index=False)

            mark = "  *BEST*" if improved else ""
            print(
                f"Epoch {epoch:02d}/{EPOCHS}"
                f" | train={train_loss:.4f} (bce={train_bce:.4f}, mse={train_mse:.4f})"
                f" | val={val_loss:.4f} (bce={val_bce:.4f}, mse={val_mse:.4f})"
                f" | best={best_val:.4f}"
                f" | lr={lr_now:.2e}"
                f" | {epoch_sec/60:.1f} min"
                f" | no_improve={epochs_no_improve}{mark}",
                flush=True,
            )

            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(
                    f"[EARLY STOP] no validation improvement for {epochs_no_improve} epoch(s). "
                    f"Stopping at epoch {epoch}.",
                    flush=True,
                )
                break

    except KeyboardInterrupt:
        print("\n[INTERRUPT] saving last checkpoint before exit...", flush=True)
        save_checkpoint(
            last_ckpt_path,
            model,
            optimizer,
            epoch=current_epoch,
            best_val=best_val,
            target_cols=TARGET_COLS,
            train_loss=float("nan"),
            val_loss=float("nan"),
        )
        print(f"[INTERRUPT] saved {last_ckpt_path}", flush=True)
        raise


if __name__ == "__main__":
    main()
