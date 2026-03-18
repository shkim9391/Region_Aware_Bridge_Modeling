from pathlib import Path
import json
import pandas as pd
from PIL import Image

BASE = Path(__file__).resolve().parent
OUT = BASE / "output"
OUT.mkdir(exist_ok=True)

PATCH_SIZE = 256

from pathlib import Path

def resolve_hires_image(spatial_dir: Path) -> Path:
    """
    Return a path to a *hires* image that matches tissue_hires_scalef coordinates.
    Priority:
      (1) outs/spatial/tissue_hires_image.*
      (2) binned spatial/tissue_hires_image.*
      (3) if only lowres exists, upsample to hires and save as binned spatial/tissue_hires_image.png
    """
    spatial_dir = spatial_dir.resolve()

    # spatial_dir = .../outs/binned_outputs/square_008um/spatial
    # outs_dir should be .../outs
    outs_dir = spatial_dir.parents[3]
    outs_spatial = outs_dir / "spatial"

    # scalefactors: prefer outs/spatial, fallback to binned spatial
    sf_path = outs_spatial / "scalefactors_json.json"
    if not sf_path.exists():
        sf_path = spatial_dir / "scalefactors_json.json"
    if not sf_path.exists():
        raise FileNotFoundError(f"Missing scalefactors_json.json in {outs_spatial} or {spatial_dir}")

    sf = json.loads(sf_path.read_text())

    # (1) Try canonical hires in outs/spatial
    for ext in ("png", "jpg", "jpeg", "tif", "tiff"):
        p = outs_spatial / f"tissue_hires_image.{ext}"
        if p.exists():
            return p

    # (2) Try hires already in binned spatial
    for ext in ("png", "jpg", "jpeg", "tif", "tiff"):
        p = spatial_dir / f"tissue_hires_image.{ext}"
        if p.exists():
            return p

    # (3) Upsample lowres -> hires and save (prefer binned spatial, fallback to OUT)
    low = None
    for ext in ("png", "jpg", "jpeg", "tif", "tiff"):
        p = spatial_dir / f"tissue_lowres_image.{ext}"
        if p.exists():
            low = p
            break
        p = outs_spatial / f"tissue_lowres_image.{ext}"
        if p.exists():
            low = p
            break

    if low is None:
        raise FileNotFoundError(
            f"No tissue_hires_image.* or tissue_lowres_image.* found in {outs_spatial} or {spatial_dir}"
        )

    r = float(sf["tissue_hires_scalef"]) / float(sf["tissue_lowres_scalef"])

    # Create the hires image in memory FIRST
    img = Image.open(low).convert("RGB")
    new_size = (int(round(img.size[0] * r)), int(round(img.size[1] * r)))
    img = img.resize(new_size, resample=Image.BICUBIC)

    # Preferred output location: binned spatial dir
    hires_out = spatial_dir / "tissue_hires_image.png"

    try:
        hires_out.parent.mkdir(parents=True, exist_ok=True)
        img.save(hires_out)
        return hires_out
    except (FileNotFoundError, PermissionError):
        # Fallback: always-writable cache in OUT/
        ds_name = spatial_dir.parents[4].name  # "CRC" or "Breast"
        hires_out2 = OUT / f"{ds_name}_tissue_hires_image.png"
        hires_out2.parent.mkdir(parents=True, exist_ok=True)
        img.save(hires_out2)
        return hires_out2
    
    # Try to write into the binned spatial dir first
    try:
        hires_out.parent.mkdir(parents=True, exist_ok=True)
        img.save(hires_out)
        return hires_out
    except (FileNotFoundError, PermissionError):
        # Fallback: always-writable cache in OUT/
        ds_name = spatial_dir.parents[4].name  # e.g., "CRC" or "Breast"
        hires_out2 = OUT / f"{ds_name}_tissue_hires_image.png"
        hires_out2.parent.mkdir(parents=True, exist_ok=True)
        img.save(hires_out2)
        return hires_out2

def norm_barcode(s: pd.Series) -> pd.Series:
    s = s.astype(str)
    s = s.str.replace("-1", "", regex=False)
    return s

def build_patch_index(dataset: str, spatial_dir: Path, z_csv: Path, out_csv: Path):
    spatial_dir = Path(spatial_dir)

    # ---- Load z_spatial ----
    z = pd.read_csv(z_csv)
    z["barcode"] = norm_barcode(z["barcode"])

    # ---- Load tissue positions (Visium HD canonical) ----
    pos = pd.read_parquet(spatial_dir / "tissue_positions.parquet")
    pos["barcode"] = norm_barcode(pos["barcode"])

    # Keep only in-tissue bins
    pos = pos[pos["in_tissue"] == 1].copy()

    # Rename for clarity
    pos = pos.rename(columns={
        "pxl_col_in_fullres": "x_px",
        "pxl_row_in_fullres": "y_px",
    })
    
    # ---- Scale fullres coordinates -> hires pixel coordinates ----
    outs_dir = spatial_dir.resolve().parents[3]         # .../outs
    outs_spatial = outs_dir / "spatial"
    
    sf_path = outs_spatial / "scalefactors_json.json"
    if not sf_path.exists():
        sf_path = spatial_dir / "scalefactors_json.json"
    
    sf = json.loads(sf_path.read_text())
    hires_sf = float(sf["tissue_hires_scalef"])
    
    pos["x_px"] = (pos["x_px"] * hires_sf).round().astype(int)
    pos["y_px"] = (pos["y_px"] * hires_sf).round().astype(int)

    # ---- Merge ----
    df = z.merge(pos[["barcode", "x_px", "y_px"]], on="barcode", how="inner")

    if len(df) == 0:
        raise RuntimeError(
            f"[{dataset}] Patch index merge produced 0 rows.\n"
            f"Example z barcode: {z['barcode'].iloc[0]}\n"
            f"Example pos barcode: {pos['barcode'].iloc[0]}"
        )

   # ---- Resolve hires image (works across 10x output variants) ----
    img_path = resolve_hires_image(spatial_dir)  # may return outs/spatial hires OR create one from lowres
    img = Image.open(img_path).convert("RGB")
    W, H = img.size

    half = PATCH_SIZE // 2

    df["x0"] = (df["x_px"] - half).clip(0, W - PATCH_SIZE).astype(int)
    df["y0"] = (df["y_px"] - half).clip(0, H - PATCH_SIZE).astype(int)
    df["x1"] = df["x0"] + PATCH_SIZE
    df["y1"] = df["y0"] + PATCH_SIZE

    df["dataset"] = dataset
    df["image_path"] = str(img_path)

    df.to_csv(out_csv, index=False)
    print(f"[OK] {dataset}: wrote {out_csv} with {len(df):,} rows")

def main():
    build_patch_index(
        "CRC",
        BASE / "CRC/outs/binned_outputs/square_008um/spatial",
        OUT / "crc_z_spatial.csv",
        OUT / "crc_patch_index.csv",
    )

    build_patch_index(
        "Breast",
        BASE / "Breast/outs/binned_outputs/square_008um/spatial",
        OUT / "breast_z_spatial.csv",
        OUT / "breast_patch_index.csv",
    )

if __name__ == "__main__":
    main()
