import scanpy as sc

crc = sc.read_h5ad("/output/crc_008um.h5ad")
br  = sc.read_h5ad("/output/breast_008um.h5ad")

print("CRC:", crc.shape, crc.obs["cluster_graphbased"].isna().sum(), "missing clusters")
print("Breast:", br.shape, br.obs["cluster_graphbased"].isna().sum(), "missing clusters")
print("CRC clusters:", crc.obs["cluster_graphbased"].nunique())
print("Breast clusters:", br.obs["cluster_graphbased"].nunique())
