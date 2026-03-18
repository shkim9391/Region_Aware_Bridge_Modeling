from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


Z_DIMS = [
    "epi_like",
    "fibroblast",
    "endothelial",
    "myeloid",
    "lymphoid_plasma",
    "smooth_myoepi",
    "stressed",
    "IFNG",
    "IFN1",
    "CYTOTOX",
    "AP_MHC",
    "NFKB",
    "PROLIF",
    "HYPOXIA",
    "EMT",
    "OXPHOS",
    "UPR",
    "ECM",
    "ANGIO",
]

# BCE / compositional-like targets
COMP_DIMS = [
    "epi_like",
    "fibroblast",
    "endothelial",
    "myeloid",
    "lymphoid_plasma",
    "smooth_myoepi",
    "stressed",
]


def split_train_val_by_blocks(
    df: pd.DataFrame,
    val_frac: float = 0.15,
    block_px: int = 2048,
    seed: int = 0,
):
    """
    Block-aware split based on x0/y0 and image_path.
    Keeps nearby spots together to reduce leakage.
    """
    if len(df) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    work = df.copy().reset_index(drop=True)

    if "image_path" in work.columns:
        img_key = work["image_path"].astype(str)
    else:
        img_key = pd.Series(["img0"] * len(work))

    bx = (pd.to_numeric(work["x0"], errors="coerce").fillna(0).astype(int) // block_px).astype(str)
    by = (pd.to_numeric(work["y0"], errors="coerce").fillna(0).astype(int) // block_px).astype(str)

    work["_block_id"] = img_key + "::" + bx + "::" + by

    block_ids = work["_block_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(block_ids)

    n_val_blocks = max(1, int(round(len(block_ids) * val_frac)))
    val_blocks = set(block_ids[:n_val_blocks])

    val_mask = work["_block_id"].isin(val_blocks).to_numpy()
    val_idx = np.where(val_mask)[0]
    train_idx = np.where(~val_mask)[0]

    return train_idx, val_idx


class WSIToZDataset(Dataset):
    def __init__(
        self,
        patch_index_csvs: Sequence[str | Path] | None = None,
        csv_paths: Sequence[str | Path] | None = None,
        z_cols: Sequence[str] = Z_DIMS,
        patch_size: int = 224,
        max_rows: int | None = None,
        seed: int = 0,
    ):
        # allow either old or new argument name
        if patch_index_csvs is None and csv_paths is None:
            raise ValueError("Provide patch_index_csvs or csv_paths.")
        if patch_index_csvs is None:
            patch_index_csvs = csv_paths

        self.z_cols = list(z_cols)
        self.patch_size = int(patch_size)
        self.rng = np.random.default_rng(seed)

        dfs = []
        for p in patch_index_csvs:
            p = Path(p)
            df = pd.read_csv(p).copy()
            dfs.append(df)

        self.df = pd.concat(dfs, ignore_index=True)

        if max_rows is not None and len(self.df) > max_rows:
            keep = self.rng.choice(len(self.df), size=max_rows, replace=False)
            self.df = self.df.iloc[np.sort(keep)].reset_index(drop=True)
        else:
            self.df = self.df.reset_index(drop=True)

        self.to_tensor = transforms.ToTensor()
        self._img_cache: dict[str, Image.Image] = {}

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, image_path: str) -> Image.Image:
        image_path = str(image_path)
        if image_path not in self._img_cache:
            self._img_cache[image_path] = Image.open(image_path).convert("RGB")
        return self._img_cache[image_path]

    def _crop_patch(self, img: Image.Image, x0: int, y0: int) -> Image.Image:
        half = self.patch_size // 2

        left = max(0, x0 - half)
        top = max(0, y0 - half)
        right = min(img.width, left + self.patch_size)
        bottom = min(img.height, top + self.patch_size)

        left = max(0, right - self.patch_size)
        top = max(0, bottom - self.patch_size)

        patch = img.crop((left, top, right, bottom))

        if patch.size != (self.patch_size, self.patch_size):
            patch = patch.resize((self.patch_size, self.patch_size), Image.BILINEAR)

        return patch

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        image_path = row["image_path"]
        x0 = int(row["x0"])
        y0 = int(row["y0"])

        img = self._load_image(image_path)
        patch = self._crop_patch(img, x0, y0)

        x = self.to_tensor(patch)

        y = torch.tensor(
            row[self.z_cols].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )

        return x, y
