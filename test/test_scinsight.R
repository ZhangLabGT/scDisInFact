rm(list = ls())
gc()
library(scINSIGHT)
library(Seurat)
setwd("/localscratch/ziqi/scDisInFact/test/")

# details check: https://github.com/Vivianstats/scINSIGHT/wiki/scINSIGHT-vignette, two sample datasets are given: real_count.rds, sim_count.rds. 
# And the scINSIGHT object of two datasets: real_scobj.rds, sim_scobj.rds.

# Load the simulated data, processed already
# sim.counts = readRDS("../data/scINSIGHT/sim_count.rds")
# write.table(sim.counts[[1]], file = "../data/scINSIGHT/GxC1.txt", sep = "\t", quote = FALSE)
# write.table(sim.counts[[2]], file = "../data/scINSIGHT/GxC2.txt", sep = "\t", quote = FALSE)
# write.table(sim.counts[[3]], file = "../data/scINSIGHT/GxC3.txt", sep = "\t", quote = FALSE)
# write.table(sim.counts[[4]], file = "../data/scINSIGHT/GxC4.txt", sep = "\t", quote = FALSE)
# write.table(sim.counts[[5]], file = "../data/scINSIGHT/GxC5.txt", sep = "\t", quote = FALSE)
# write.table(sim.counts[[6]], file = "../data/scINSIGHT/GxC6.txt", sep = "\t", quote = FALSE)

# sim.sample = c("S1", "S2", "S3", "S4", "S5", "S6")
# # Name of count list should represent sample names
# names(sim.counts) = sim.sample
# sim.condition = c("T1", "T1", "T2", "T2", "T3", "T3")
# # Name of condition vector should match sample names in count list
# names(sim.condition) = sim.sample
# 
# # Create an scINSIGHT object 
# sim.scobj = create_scINSIGHT(norm.data = sim.counts, condition = sim.condition)
# 
# sim.scobj = run_scINSIGHT(sim.scobj, K = seq(5,15,2), out.dir = './scinsight_intermid/',
#                           num.cores = 5)

# sim.scobj = readRDS("../data/scINSIGHT/sim_scobj.rds")
# print(sim.scobj@parameters)

# # V is a matrix whose dimension is K by gene number
# print(dim(sim.scobj@V))
# # H is a list whose length equals the condition number
# print(length(sim.scobj@H))

# # Sample name of each single cell
# sample = unlist(lapply(1:length(sim.counts), function(i){
#   rep(sim.sample[i], ncol(sim.counts[[i]]))
# }))
# # Condition of each single cell
# condition = unlist(lapply(1:length(sim.counts), function(i){
#   rep(sim.condition[i], ncol(sim.counts[[i]]))
# }))
# # True cell type of each single cell
# celltype = unlist(lapply(1:length(sim.counts), function(i){
#   lapply(1:ncol(sim.counts[[i]]), function(j){
#     toupper(unlist(strsplit(colnames(sim.counts[[i]])[[j]], "[.]"))[[1]])})}))

# # Cluster results of each single cell
# clusters = as.character(unlist(sim.scobj@clusters))
# 
# library(Rtsne)
# # W2 is the normalized expression levels of common gene modules
# W2 = Reduce(rbind, sim.scobj@norm.W_2)
# set.seed(1)
# res_tsne = Rtsne(W2, dims = 2, check_duplicates = FALSE)
# 
# da_tsne = data.frame(tSNE1 = res_tsne$Y[,1], tSNE2 = res_tsne$Y[,2],
#                      sample = sample, condition = condition,
#                      celltype = celltype, clusters = clusters)



################################################################################
#
# Symsim datasets
#
################################################################################
dataset <- "imputation_10000_500_0.4_50_2"
data_dir <- paste0("../data/simulated_new/", dataset,"/")
result_dir <- paste0("./simulated/imputation_new/", dataset, "/")
dir.create(file.path(result_dir, "scinsight_2/"), showWarnings = FALSE)

GxC1 <- as.matrix(read.table(paste0(data_dir, "GxC1_ctrl.txt"), sep = "\t"))
rownames(GxC1) <- paste("gene-", seq(1, dim(GxC1)[1]), sep = "")
colnames(GxC1) <- paste("cell-", seq(1, dim(GxC1)[2]), sep = "")
GxC2 <- as.matrix(read.table(paste0(data_dir, "GxC2_ctrl.txt"), sep = "\t"))
rownames(GxC2) <- paste("gene-", seq(1, dim(GxC2)[1]), sep = "")
colnames(GxC2) <- paste("cell-", seq(1, dim(GxC2)[2]), sep = "")
GxC3 <- as.matrix(read.table(paste0(data_dir, "GxC3_stim1.txt"), sep = "\t"))
rownames(GxC3) <- paste("gene-", seq(1, dim(GxC3)[1]), sep = "")
colnames(GxC3) <- paste("cell-", seq(1, dim(GxC3)[2]), sep = "")
GxC4 <- as.matrix(read.table(paste0(data_dir, "GxC4_stim1.txt"), sep = "\t"))
rownames(GxC4) <- paste("gene-", seq(1, dim(GxC4)[1]), sep = "")
colnames(GxC4) <- paste("cell-", seq(1, dim(GxC4)[2]), sep = "")
GxC5 <- as.matrix(read.table(paste0(data_dir, "GxC5_stim2.txt"), sep = "\t"))
rownames(GxC5) <- paste("gene-", seq(1, dim(GxC5)[1]), sep = "")
colnames(GxC5) <- paste("cell-", seq(1, dim(GxC5)[2]), sep = "")
GxC6 <- as.matrix(read.table(paste0(data_dir, "GxC6_stim2.txt"), sep = "\t"))
rownames(GxC6) <- paste("gene-", seq(1, dim(GxC6)[1]), sep = "")
colnames(GxC6) <- paste("cell-", seq(1, dim(GxC6)[2]), sep = "")

sim.counts <- list(GxC1, GxC2, GxC3, GxC4, GxC5, GxC6)

# normalization
sim.counts <- lapply(1:length(sim.counts), function(i){
  # Initialize the Seurat object with the raw data
  x = CreateSeuratObject(counts = sim.counts[[i]], assay = "RNA")
  # Normalize the data 
  x = NormalizeData(x)
  # no filtering step
  return(x)
})
sim.counts <- lapply(sim.counts, function(x){
  as.matrix(x@assays$RNA@data)
})


sim.sample <- c("S1", "S2", "S3", "S4", "S5", "S6")
# Name of count list should represent sample names
names(sim.counts) <- sim.sample
sim.condition  <- c("ctrl", "ctrl", "stim1", "stim1", "stim2", "stim2")
# Name of condition vector should match sample names in count list
names(sim.condition) <- sim.sample

# Create an scINSIGHT object
sim.scobj <- create_scINSIGHT(norm.data = sim.counts, condition = sim.condition)
# K_j default is 2
sim.scobj <- run_scINSIGHT(sim.scobj, K = seq(5,15,2), K_j = 2, out.dir = './scinsight_intermid/', num.cores = 4)
# save RDS file
saveRDS(sim.scobj, file = paste0(result_dir, "scinsight_2.rds"))

# read RDS file
sim.scobj <- readRDS(paste0(result_dir, "scinsight_2.rds"))

# for key gene discovery
Hs <- lapply(1:length(sim.scobj@H), function(i){
  H <- sim.scobj@H[[i]]
  write.table(H, paste0(result_dir, "scinsight_2/H_", i, ".txt"), sep = "\t", quote = FALSE, )
  return(H)
})

W1s <- lapply(1:length(sim.scobj@W_1), function(i){
  W1 <- sim.scobj@W_1[[i]]
  write.table(W1, paste0(result_dir, "scinsight_2/W1", i, ".txt"), sep = "\t", quote = FALSE, )
  return(W1)
})

# for cell embedding visualization
# norm.W_2s <- lapply(1:length(sim.scobj@norm.W_2), function(i){
#   W_2 <- sim.scobj@norm.W_2[[i]]
#   write.table(W_2, paste0(result_dir, "scinsight/W2_", i, ".txt"), sep = "\t")
#   return(W_2)
# })
W2 <- Reduce(rbind, sim.scobj@norm.W_2)
write.table(W2, paste0(result_dir, "scinsight_2/W2.txt"), sep = "\t", quote = FALSE)


# # V is a matrix whose dimension is K by gene number
# print(dim(sim.scobj@V))
# # H is a list whose length equals the condition number
# print(length(sim.scobj@H))

# # Sample name of each single cell
# sample = unlist(lapply(1:length(sim.counts), function(i){
#   rep(sim.sample[i], ncol(sim.counts[[i]]))
# }))
# # Condition of each single cell
# condition = unlist(lapply(1:length(sim.counts), function(i){
#   rep(sim.condition[i], ncol(sim.counts[[i]]))
# }))
# # True cell type of each single cell
# celltype = unlist(lapply(1:length(sim.counts), function(i){
#   lapply(1:ncol(sim.counts[[i]]), function(j){
#     toupper(unlist(strsplit(colnames(sim.counts[[i]])[[j]], "[.]"))[[1]])})}))

# # Cluster results of each single cell
# clusters = as.character(unlist(sim.scobj@clusters))
# 
# library(Rtsne)
# # W2 is the normalized expression levels of common gene modules
# W2 = Reduce(rbind, sim.scobj@norm.W_2)
# set.seed(1)
# res_tsne = Rtsne(W2, dims = 2, check_duplicates = FALSE)
# 
# da_tsne = data.frame(tSNE1 = res_tsne$Y[,1], tSNE2 = res_tsne$Y[,2],
#                      sample = sample, condition = condition,
#                      celltype = celltype, clusters = clusters)



