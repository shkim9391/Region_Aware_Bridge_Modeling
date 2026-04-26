import scanpy as sc
import pandas as pd
from pandas.api.types import CategoricalDtype
from pathlib import Path

BASE = Path(__file__).resolve().parent
crc = sc.read_h5ad(str(BASE / "output" / "crc_008um.h5ad"))
br  = sc.read_h5ad(str(BASE / "output" / "breast_008um.h5ad"))

for ad in (crc, br):
    s = ad.obs["cluster_graphbased"]
    if not isinstance(s.dtype, CategoricalDtype):
        s = s.astype("category")
    if "Background" not in s.cat.categories:
        s = s.cat.add_categories(["Background"])
    ad.obs["cluster_graphbased"] = s.fillna("Background")

print("CRC Background bins:", (crc.obs["cluster_graphbased"] == "Background").sum())
print("Breast Background bins:", (br.obs["cluster_graphbased"] == "Background").sum())
