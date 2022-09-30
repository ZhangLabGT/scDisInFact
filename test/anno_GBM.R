# read raw counts
rm(list = ls())
gc()
library(reticulate)
library(Seurat)
library(Matrix)
sp <- import("scipy.sparse")

setwd("/localscratch/ziqi/scDisInFact/test/")
data_dir <- "../data/GBM_treatment/Fig4/processed/"
genes <- read.table(paste0(data_dir, "genes.txt"))[[1]]
meta.cells <- read.csv(paste0(data_dir, "meta_cells.csv"), sep = "\t", row.names = 1)
count.rna <- sp$load_npz(paste0(data_dir, "counts_rna_csc.npz"))
colnames(count.rna) <- genes
rownames(count.rna) <- rownames(meta.cells)
count.rna <- t(count.rna)
# create seurat object
gbm <- CreateSeuratObject(counts = count.rna, project = "GBM")
gbm <- AddMetaData(gbm, meta.cells)


#############################################################################################
#
# Integrate and annotate
#
#############################################################################################

# split the dataset into a list of two seurat objects (stim and CTRL)
gbm.list <- SplitObject(gbm, split.by = "sample_id")

# normalize and identify variable features for each dataset independently
gbm.list <- lapply(X = gbm.list, FUN = function(x) {
  x <- NormalizeData(x)
  x <- FindVariableFeatures(x, selection.method = "vst", nfeatures = 2000)
})

# select features that are repeatedly variable across datasets for integration
features <- SelectIntegrationFeatures(object.list = gbm.list)
# integration
gbm.anchors <- FindIntegrationAnchors(object.list = gbm.list, anchor.features = features)
gbm.combined <- IntegrateData(anchorset = gbm.anchors)

# specify that we will perform downstream analysis on the corrected data note that the
# original unmodified data still resides in the 'RNA' assay
DefaultAssay(gbm.combined) <- "integrated"

# Run the standard workflow for visualization and clustering
gbm.combined <- ScaleData(gbm.combined, verbose = FALSE)
gbm.combined <- RunPCA(gbm.combined, npcs = 30, verbose = FALSE)
gbm.combined <- RunUMAP(gbm.combined, reduction = "pca", dims = 1:30)
gbm.combined <- FindNeighbors(gbm.combined, reduction = "pca", dims = 1:30)
gbm.combined <- FindClusters(gbm.combined, resolution = 0.5)


saveRDS(gbm.combined, paste0(data_dir, "seurat_int.rds"))
gbm.combind <- readRDS(paste0(data_dir, "seurat_int.rds"))

# Plots
# DimPlot(object = gbm.combined, reduction = "umap", pt.size = 0.1,label = FALSE, group.by = "sample_id")
# DimPlot(object = gbm.combined, reduction = "umap", pt.size = 0.1,label = FALSE, group.by = "patient_id")
# DimPlot(object = gbm.combined, reduction = "umap", pt.size = 0.1,label = FALSE, group.by = "treatment")
# DimPlot(object = gbm.combined, reduction = "umap", pt.size = 0.1,label = FALSE, group.by = "mstatus")
# DimPlot(object = gbm.combined, reduction = "umap", label = TRUE, repel = TRUE)

# # check the distribution of markers
# FeaturePlot(gbm.combined, features = c("CD3D", "SELL", "CREM", "CD8A", "GNLY", "CD79A", "FCGR3A", "CCL2", "PPBP"), min.cutoff = "q9")
# # rename the cell type by markers
# gbm.combined <- RenameIdents(gbm.combined, `0` = "CD14 Mono", `1` = "CD4 Naive T", `2` = "CD4 Memory T",
#                                 `3` = "CD16 Mono", `4` = "B", `5` = "CD8 T", `6` = "NK", `7` = "T activated", `8` = "DC", `9` = "B Activated",
#                                 `10` = "Mk", `11` = "pDC", `12` = "Eryth", `13` = "Mono/Mk Doublets", `14` = "HSPC")
# DimPlot(gbm.combined, label = TRUE)

#############################################################################################
#
# Annotate batch by batch
#
#############################################################################################

# split the dataset into a list of two seurat objects (stim and CTRL)
gbm.list <- SplitObject(gbm, split.by = "sample_id")

# normalize and identify variable features for each dataset independently
gbm.list <- lapply(X = gbm.list, FUN = function(x) {
  x <- NormalizeData(x)
  all.genes<-rownames(x)
  x <- ScaleData(x, features = all.genes)
  x <- FindVariableFeatures(x, selection.method = "vst", nfeatures = 2000)
  x <- RunPCA(x,npcs = 30, ndims.print = 1:5)
  x <- RunUMAP(object = x, dims = 1:30)
  x <- FindNeighbors(x, reduction = "pca", dims = 1:30)
  x <- FindClusters(x, resolution = 0.5)
})

gbm.sample <- gbm.list[[5]]

# cluster results
DimPlot(object = gbm.sample, reduction = "umap", label = TRUE, repel = TRUE)
# m status
# DimPlot(object = gbm.sample, reduction = "umap", pt.size = 0.1,label = FALSE, group.by = "mstatus")

# check the distribution of markers
# Myeloid markers: CD14, AIF1, TYROBP, CD163
# Oligodendrocytes: PLP1, MBP, MAG, SOX10
# T cell: TRAC, TRBC1, TRBC2, CD3D
# Endothelial cell: ESM1, ITM2A, VWF, CLDN5
# Pericytes: PDGFRB, DCN, COL3A1, RGS5 

# there exists all zero genes, better to use the combined dataset after integration
# Myeloid: 3
FeaturePlot(gbm.sample, features = c("ENSG00000170458.13-CD14", "ENSG00000204472.12-AIF1", "ENSG00000011600.11-TYROBP", "ENSG00000177575.12-CD163"), min.cutoff = "q9")

# Oligodendrocytes: 1
FeaturePlot(gbm.sample, features = c("ENSG00000123560.13-PLP1", "ENSG00000197971.14-MBP", "ENSG00000105695.14-MAG", "ENSG00000100146.16-SOX10"), min.cutoff = "q9")

# T cell: 5
FeaturePlot(gbm.sample, features = c("ENSG00000211751.8-TRBC1", "ENSG00000211772.9-TRBC2", "ENSG00000277734.5-TRAC", "ENSG00000167286.9-CD3D"), min.cutoff = "q9")

# Endothelial
FeaturePlot(gbm.sample, features = c("ENSG00000164283.12-ESM1", "ENSG00000078596.10-ITM2A", "ENSG00000110799.13-VWF", "ENSG00000184113.9-CLDN5"), min.cutoff = "q9")

# Pericytes
FeaturePlot(gbm.sample, features = c("ENSG00000113721.13-PDGFRB", "ENSG00000011465.16-DCN", "ENSG00000168542.12-COL3A1", "ENSG00000232995.7-RGS5"), min.cutoff = "q9")


# rename the cell type by markers: sample 1
# gbm.sample <- RenameIdents(gbm.sample, `2` = "Oligodendrocytes", `4` = "Myeloid", `0` = "Other", `3` = "Other", `1` = "Other", `5` = "Other", `6` = "Other")
# sample 2
# gbm.sample <- RenameIdents(gbm.sample, `2` = "Oligodendrocytes", `1` = "Myeloid", `6` = "Myeloid", `7` = "Myeloid", `9` = "T cell", `10` = "Pericytes", `0` = "Other", `3` = "Other", `4` = "Other", `5` = "Other", `8` = "Other")
# sample 3
# gbm.sample <- RenameIdents(gbm.sample, `3` = "Oligodendrocytes", `2` = "Myeloid", `6` = "Myeloid", `7` = "Myeloid", `9` = "Myeloid", `1` = "Other", `0` = "Other", `4` = "Other", `5` = "Other", `8` = "Other")
# sample 4
# gbm.sample <- RenameIdents(gbm.sample, `0` = "Other", `1` = "Myeloid", `2` = "Other", `3` = "Other", `4` = "Oligodendrocytes", `5` = "Other", `6` = "Other", `7` = "Oligodendrocytes", `8` = "T cell", `9` = "Other")
# sample 5
# gbm.sample <- RenameIdents(gbm.sample, `0` = "Other", `1` = "Myeloid", `2` = "Other", `3` = "Oligodendrocytes", `4` = "Myeloid", `5` = "Myeloid", `6` = "Other", `7` = "Myeloid", `8` = "Other", `9` = "T cell", `10` = "Other", `11` = "Other")
# sample 6
# gbm.sample <- RenameIdents(gbm.sample, `0` = "Myeloid", `1` = "Other", `2` = "Other", `3` = "Other", `4` = "Oligodendrocytes", `5` = "Other", `6` = "Other", `7` = "T cell", `8` = "Myeloid")
# sample 7
# gbm.sample <- RenameIdents(gbm.sample, `0` = "Other", `1` = "Other", `2` = "Myeloid", `3` = "Other", `4` = "Other", `5` = "Oligodendrocytes")
# sample 8
# gbm.sample <- RenameIdents(gbm.sample, `0` = "Other", `1` = "Other", `2` = "Myeloid", `3` = "Oligodendrocytes")
# sample 9
# gbm.sample <- RenameIdents(gbm.sample, `0` = "Other", `1` = "Other", `2` = "Other", `3` = "Other", `4` = "Oligodendrocytes", `5` = "Other", `6` = "Other", `7` = "Other", `8` = "Other", `9` = "Myeloid", `10` = "Other")
# sample 10
# gbm.sample <- RenameIdents(gbm.sample, `0` = "Other", `1` = "Other", `2` = "Other", `3` = "Other", `4` = "Other", `5` = "Other", `6` = "Other", `7` = "Other", `8` = "Myeloid", `9` = "Oligodendrocytes", `10` = "Myeloid", `11` = "Other")
# sample 11
# gbm.sample <- RenameIdents(gbm.sample, `0` = "Other", `1` = "Other", `2` = "Other", `3` = "Oligodendrocytes", `4` = "Myeloid", `5` = "Other", `6` = "Other", `7` = "Other", `8` = "Other")
# sample 12
# gbm.sample <- RenameIdents(gbm.sample, `0` = "Oligodendrocytes", `1` = "Myeloid", `2` = "Other", `3` = "Other", `4` = "Other", `5` = "Other", `6` = "Other", `7` = "Other")
# sample 13
# gbm.sample <- RenameIdents(gbm.sample, `0` = "Other", `1` = "Other", `2` = "Myeloid", `3` = "Other", `4` = "Oligodendrocytes", `5` = "Other", `6` = "Other", `7` = "Other", `8` = "Endothelial")
# sample 14
# gbm.sample <- RenameIdents(gbm.sample, `0` = "Other", `1` = "Other", `2` = "Myeloid", `3` = "Other", `4` = "Oligodendrocytes", `5` = "Other", `6` = "Other", `7` = "Other")
# sample 15
# gbm.sample <- RenameIdents(gbm.sample, `0` = "Other", `1` = "Oligodendrocytes", `2` = "Other", `3` = "Myeloid", `4` = "Other")
# sample 16
# gbm.sample <- RenameIdents(gbm.sample, `0` = "Other", `1` = "Other", `2` = "Oligodendrocytes", `3` = "Other", `4` = "Myeloid", `5` = "T cell", `6` = "Other")
# sample 17
# gbm.sample <- RenameIdents(gbm.sample, `0` = "Other", `1` = "Oligodendrocytes", `2` = "Oligodendrocytes", `3` = "Other", `4` = "T cell", `5` = "Myeloid", `6` = "Other")
# sample 18
# gbm.sample <- RenameIdents(gbm.sample, `0` = "Other", `1` = "Myeloid", `2` = "Oligodendrocytes", `3` = "Other")
# sample 19
# gbm.sample <- RenameIdents(gbm.sample, `0` = "Other", `1` = "Oligodendrocytes", `2` = "Oligodendrocytes", `3` = "Myeloid")
# sample 20
# gbm.sample <- RenameIdents(gbm.sample, `0` = "Other", `1` = "Oligodendrocytes", `2` = "Myeloid", `3` = "Oligodendrocytes")
# sample 21
gbm.sample <- RenameIdents(gbm.sample, `0` = "Other", `1` = "Oligodendrocytes", `2` = "Other", `3` = "Myeloid", `4` = "Other", `5` = "T cell")


gbm.sample@meta.data[gbm.sample@meta.data[["mstatus"]] == "non-tumor", "mstatus"] <- as.character(gbm.sample@active.ident[gbm.sample@meta.data[["mstatus"]] == "non-tumor"])
DimPlot(object = gbm.sample, reduction = "umap", pt.size = 0.1,label = FALSE, group.by = "mstatus")

meta.sample <- gbm.sample@meta.data


# Combined meta.samples
gbm@meta.data[rownames(meta.sample), "mstatus"] <- meta.sample[,"mstatus"]

# write.table(gbm@meta.data, file = paste0(data_dir, "meta_cells_seurat.csv"), sep = "\t", quote = FALSE)



