rm(list = ls())
gc()
# library("devtools")
# devtools::install_github('YosefLab/SymSim')
library("SymSim")
setwd("/localscratch/ziqi/scDisInFact/data/simulated/")


#-----------------------------------------------------------------------------------
#
# Simulation parameter setting
#
#-----------------------------------------------------------------------------------
# Simulate multiple discrete populations
# When there are multiple populations, users need to provide a tree. A tree with five leaves (five populations) can be generated as follows:
# phyla1 <- read.tree(text="((A:1,B:1,E:1.5):1,(C:0.5,D:0.5):1.5);")
# here we use a more complicated tree to generate more cell clusters
phyla1 <- read.tree(text="(((A:1,B:1,F:1.5):1,(C:0.5,D:0.5,G:1.5):1.5,(H:0.5,I:0.5,J:1.5):2.0):1,((K:1,L:1,M:1.5):2.5,(N:0.5,O:0.5,P:1.5):3.0):2);")
# total number of genes
ngenes <- 500
# total number of cells
ncells_total <- 10000
# the size of the smallest cell cluster
min_popsize <- 500
# cluster separation, larger sigma gives cell clusters that are more mixed together
Sigma <- 0.4
# number of cell batches
nbatch <- 2
# perturbation parameters
epsilon <- 2
# number of perturbed genes
n_diff_genes <- 50
# simulation function that generate the true count (without condition and batch effect)
true_counts_res <- SimulateTrueCounts(ncells_total=ncells_total, min_popsize=min_popsize, i_minpop=2, ngenes=ngenes, nevf=10, evf_type="discrete", n_de_evf=9, vary="s", Sigma=Sigma, phyla=phyla1, randseed=0)
true_counts_res_dis <- true_counts_res
tsne_true_counts <- PlotTsne(meta=true_counts_res[[3]], data=log2(true_counts_res[[1]]+1), evf_type="discrete", n_pc=20, label='pop', saving = F, plotname="discrete populations (true counts)")
tsne_true_counts[[2]]
# write.table(true_counts_res[[1]], file = "true_counts.csv", quote = FALSE, sep = ",", row.names = FALSE, col.names = FALSE)
# write.table(true_counts_res[[3]]["pop"], file = "anno.csv", quote = FALSE, row.names = FALSE, col.names = FALSE)


#-----------------------------------------------------------------------------------
#
# Generate observed count with batch effect
#
#-----------------------------------------------------------------------------------
# transform to observed counts, using UMI count
data(gene_len_pool)
gene_len <- sample(gene_len_pool, ngenes, replace = FALSE)
# add sequencing noise
observed_counts <- True2ObservedCounts(true_counts=true_counts_res[[1]], meta_cell=true_counts_res[[3]], protocol="UMI", alpha_mean=0.05, alpha_sd=0.02, gene_len=gene_len, depth_mean=5e4, depth_sd=3e3)
tsne_UMI_counts <- PlotTsne(meta=observed_counts[[2]], data=log2(observed_counts[[1]]+1), evf_type="discrete", n_pc=20, label='pop', saving = F, plotname="observed counts UMI")
tsne_UMI_counts[[2]]
# add batch effect
observed_rnaseq_loBE <- DivideBatches(observed_counts_res = observed_counts, nbatch = nbatch, batch_effect_size = 1)
tsne_batches <- PlotTsne(meta=observed_rnaseq_loBE[[2]], data=log2(observed_rnaseq_loBE[[1]]+1), evf_type="discrete", n_pc=20, label='batch', saving = F, plotname="observed counts in batches")
tsne_batches[[2]]


#-----------------------------------------------------------------------------------
#
# Add condition effect
#
#-----------------------------------------------------------------------------------
set.seed(1)
cellset_batch1 <- which(observed_rnaseq_loBE[[2]]$batch==1)
cellset_batch2 <- which(observed_rnaseq_loBE[[2]]$batch==2)
# true counts
counts_batch1 <- true_counts_res[[1]][, cellset_batch1]
counts_batch2 <- true_counts_res[[1]][, cellset_batch2]
counts_true <- list(counts_batch1, counts_batch2)
meta.cells_batch1 <- true_counts_res$cell_meta[cellset_batch1,1:2]
meta.cells_batch2 <- true_counts_res$cell_meta[cellset_batch2,1:2]
meta.cells_ctrl <- list(meta.cells_batch1, meta.cells_batch2)
# control group
counts_batch1_ctrl <- observed_rnaseq_loBE[[1]][, cellset_batch1]
counts_batch2_ctrl <- observed_rnaseq_loBE[[1]][, cellset_batch2]
counts_ctrl <- list(counts_batch1_ctrl, counts_batch2_ctrl)
# control + severe symptom
interval<-1
counts_stim<-list()
meta.cells_stim <- list()
for(batch.id in seq(1,2)){
  counts_stim[[batch.id]] <- counts_ctrl[[batch.id]]
  counts_stim[[batch.id]][1:n_diff_genes,] <- counts_stim[[batch.id]][1:n_diff_genes,] + matrix(runif(n_diff_genes * dim(counts_stim[[batch.id]])[2], min = -interval, max = 0) + epsilon, n_diff_genes, dim(counts_stim[[batch.id]])[2])
  # create rare cell types only in stim condition
  # select a sub-cluster of cells, and perturb them more
  rare.cell.index <- which(meta.cells_ctrl[[batch.id]]$pop == 1)[1:200]
  counts_stim[[batch.id]][1:n_diff_genes,rare.cell.index] <- counts_stim[[batch.id]][1:n_diff_genes,rare.cell.index] + matrix(runif(n_diff_genes * length(rare.cell.index), min = -interval, max = 0) + 2*epsilon, n_diff_genes, length(rare.cell.index))
  # update meta.cells to include the rare cell types
  meta.cells_stim[[batch.id]] <- meta.cells_ctrl[[batch.id]]
  meta.cells_stim[[batch.id]][rare.cell.index, "pop"] <- 16
}

#-----------------------------------------------------------------------------------
#
# Save the data
#
#-----------------------------------------------------------------------------------
datapath <- "simulated_rare_celltype"
system(sprintf("mkdir %s", datapath))
write.table(counts_true[[1]], sprintf("%s/GxC1_true.txt", datapath), quote=F, row.names = F, col.names = F, sep = "\t")
write.table(counts_true[[2]], sprintf("%s/GxC2_true.txt", datapath), quote=F, row.names = F, col.names = F, sep = "\t")

write.table(counts_ctrl[[1]], sprintf("%s/GxC1_ctrl.txt", datapath), quote=F, row.names = F, col.names = F, sep = "\t")
write.table(counts_ctrl[[2]], sprintf("%s/GxC2_ctrl.txt", datapath), quote=F, row.names = F, col.names = F, sep = "\t")

write.table(counts_stim[[1]], sprintf("%s/GxC1_stim.txt", datapath), quote=F, row.names = F, col.names = F, sep = "\t")
write.table(counts_stim[[2]], sprintf("%s/GxC2_stim.txt", datapath), quote=F, row.names = F, col.names = F, sep = "\t")

write.table(meta.cells_ctrl[[1]], sprintf("%s/cell_label_ctrl1.txt", datapath), quote=F, row.names = F, col.names = T, sep = "\t")
write.table(meta.cells_ctrl[[2]], sprintf("%s/cell_label_ctrl2.txt", datapath), quote=F, row.names = F, col.names = T, sep = "\t")
write.table(meta.cells_stim[[1]], sprintf("%s/cell_label_stim1.txt", datapath), quote=F, row.names = F, col.names = T, sep = "\t")
write.table(meta.cells_stim[[2]], sprintf("%s/cell_label_stim2.txt", datapath), quote=F, row.names = F, col.names = T, sep = "\t")

counts <- cbind(counts_ctrl[[1]], counts_ctrl[[2]])

# Plot batches and clusters
tsne_batches <- PlotTsne(meta=observed_rnaseq_loBE[[2]][c(cellset_batch1, cellset_batch2),], data=log2(counts + 1), evf_type="discrete", n_pc=20, label='batch', saving = F, plotname="observed counts  batches (ctrl)")
tsne_clusters <- PlotTsne(meta=observed_rnaseq_loBE[[2]][c(cellset_batch1, cellset_batch2),], data=log2(counts + 1), evf_type="discrete", n_pc=20, label='pop', saving = F, plotname="observed counts clusters (ctrl)")

pdf(file=sprintf("%s/tsne.pdf", datapath))
print(tsne_batches[[2]])
print(tsne_clusters[[2]])
dev.off()
