rm(list = ls())
gc()
# library("devtools")
# devtools::install_github('YosefLab/SymSim')
library("SymSim")
setwd("/localscratch/ziqi/scDisInFact/data/simulated/unif/")

#-----------------------------------------------------------------------------------
#
# Simulation parameter setting 
#
#-----------------------------------------------------------------------------------
# Simulate multiple discrete populations
phyla1 <- read.tree(text="(((A:1,B:1,F:1.5):1,(C:0.5,D:0.5,G:1.5):1.5,(H:0.5,I:0.5,J:1.5):2.0):1,((K:1,L:1,M:1.5):2.5,(N:0.5,O:0.5,P:1.5):3.0):2,E:3);")

# total number of genes
ngenes <- 500
# total number of cells
ncells_total <- 10000
# the size of the smallest cell cluster
min_popsize <- 100
# cluster separation, larger sigma gives cell clusters that are more mixed together
Sigma <- 0.4
# number of cell batches
nbatch <- 2
# perturbation parameters
epsilon <- 8
# number of perturbed genes
n_diff_genes <- 100

# simulation function that generate the true count (without condition and batch effect)
true_counts_res <- SimulateTrueCounts(ncells_total=ncells_total, min_popsize=min_popsize, i_minpop=2, ngenes=ngenes, nevf=10, evf_type="discrete", n_de_evf=9, vary="s", Sigma=Sigma, phyla=phyla1, randseed=0)
true_counts_res_dis <- true_counts_res
tsne_true_counts <- PlotTsne(meta=true_counts_res[[3]], data=log2(true_counts_res[[1]]+1), evf_type="discrete", n_pc=20, label='pop', saving = F, plotname="discrete populations (true counts)")
tsne_true_counts[[2]]

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
# Add condition effect, 2 condition types: 
# (ctrl + healthy), (ctrl + mild), (ctrl + severe), (stim + healthy), (stim + mild), (stim + severe) 
#
#-----------------------------------------------------------------------------------
# set random seed
set.seed(1)

cellset_batch1 <- which(observed_rnaseq_loBE[[2]]$batch==1)
cellset_batch2 <- which(observed_rnaseq_loBE[[2]]$batch==2)

# true counts
counts_batch1 <- true_counts_res[[1]][, cellset_batch1]
counts_batch2 <- true_counts_res[[1]][, cellset_batch2]
counts_true <- list(counts_batch1, counts_batch2)

# control + healthy
counts_batch1_ctrl_healthy <- observed_rnaseq_loBE[[1]][, cellset_batch1]
counts_batch2_ctrl_healthy <- observed_rnaseq_loBE[[1]][, cellset_batch2]
counts_ctrl_healthy <- list(counts_batch1_ctrl_healthy, counts_batch2_ctrl_healthy)

# control + severe symptom
interval<-1
counts_ctrl_severe <- lapply(counts_ctrl_healthy, function(x){
  # use uniform distribution
  x[1:n_diff_genes,] <- x[1:n_diff_genes,] + matrix(runif(n_diff_genes * dim(x)[2], min = -interval, max = 0) + epsilon, n_diff_genes, dim(x)[2])

  # use Gaussian distribution, Guarantee positive
  # x[1:n_diff_genes,] <- 2^(log2(x[1:n_diff_genes,]) + matrix(rnorm(n_diff_genes * dim(x)[2], mean = 0, sd = 0.1) + epsilon, n_diff_genes, dim(x)[2]))
  
  return(x)
})

# stimulation + healthy symptom
counts_stim_healthy <- lapply(counts_ctrl_healthy, function(x){
  # use uniform distribution
  x[(n_diff_genes+1):(2*n_diff_genes),] <- x[(n_diff_genes+1):(2*n_diff_genes),] + matrix(runif(n_diff_genes * dim(x)[2], min = -interval, max = 0) + epsilon, n_diff_genes, dim(x)[2])
  
  # use Gaussian distribution
  # x[(n_diff_genes+1):(2*n_diff_genes),] <- 2^(log2(x[(n_diff_genes+1):(2*n_diff_genes),]) + matrix(rnorm(n_diff_genes * dim(x)[2], mean = 0, sd = 0.1) + epsilon, n_diff_genes, dim(x)[2]))

  return(x)
})

# stimulation + severe symptom
counts_stim_severe <- lapply(counts_stim_healthy, function(x){
  # use uniform distribution
  x[1:n_diff_genes,] <- x[1:n_diff_genes,] + matrix(runif(n_diff_genes * dim(x)[2], min = -interval, max = 0) + epsilon, n_diff_genes, dim(x)[2])
  
  # use Gaussian distribution
  # x[1:n_diff_genes,] <- 2^(log2(x[1:n_diff_genes,]) + matrix(rnorm(n_diff_genes * dim(x)[2], mean = 0, sd = 0.1) + epsilon, n_diff_genes, dim(x)[2]))
  
  return(x)
})

print(paste0("2conds_base_", ncells_total, "_", ngenes, "_", Sigma, "_", n_diff_genes, "_", epsilon))
datapath <- paste0("2conds_base_", ncells_total, "_", ngenes, "_", Sigma, "_", n_diff_genes, "_", epsilon)
system(sprintf("mkdir -p %s", datapath))
write.table(counts_true[[1]], sprintf("%s/GxC1_true.txt", datapath), quote=F, row.names = F, col.names = F, sep = "\t")
write.table(counts_true[[2]], sprintf("%s/GxC2_true.txt", datapath), quote=F, row.names = F, col.names = F, sep = "\t")

write.table(counts_ctrl_healthy[[1]], sprintf("%s/GxC1_ctrl_healthy.txt", datapath), quote=F, row.names = F, col.names = F, sep = "\t")
write.table(counts_ctrl_healthy[[2]], sprintf("%s/GxC2_ctrl_healthy.txt", datapath), quote=F, row.names = F, col.names = F, sep = "\t")

write.table(counts_stim_healthy[[1]], sprintf("%s/GxC1_stim_healthy.txt", datapath), quote=F, row.names = F, col.names = F, sep = "\t")
write.table(counts_stim_healthy[[2]], sprintf("%s/GxC2_stim_healthy.txt", datapath), quote=F, row.names = F, col.names = F, sep = "\t")

write.table(counts_ctrl_severe[[1]], sprintf("%s/GxC1_ctrl_severe.txt", datapath), quote=F, row.names = F, col.names = F, sep = "\t")
write.table(counts_ctrl_severe[[2]], sprintf("%s/GxC2_ctrl_severe.txt", datapath), quote=F, row.names = F, col.names = F, sep = "\t")

write.table(counts_stim_severe[[1]], sprintf("%s/GxC1_stim_severe.txt", datapath), quote=F, row.names = F, col.names = F, sep = "\t")
write.table(counts_stim_severe[[2]], sprintf("%s/GxC2_stim_severe.txt", datapath), quote=F, row.names = F, col.names = F, sep = "\t")

write.table(true_counts_res$cell_meta[cellset_batch1,1:2], sprintf("%s/cell_label1.txt", datapath), quote=F, row.names = F, col.names = T, sep = "\t")
write.table(true_counts_res$cell_meta[cellset_batch2,1:2], sprintf("%s/cell_label2.txt", datapath), quote=F, row.names = F, col.names = T, sep = "\t")



counts <- cbind(counts_ctrl_healthy[[1]], counts_ctrl_healthy[[2]])

# Plot batches and clusters
tsne_batches <- PlotTsne(meta=observed_rnaseq_loBE[[2]][c(cellset_batch1, cellset_batch2),], data=log2(counts + 1), evf_type="discrete", n_pc=20, label='batch', saving = F, plotname="observed, batches (healthy ctrl)")
tsne_clusters <- PlotTsne(meta=observed_rnaseq_loBE[[2]][c(cellset_batch1, cellset_batch2),], data=log2(counts + 1), evf_type="discrete", n_pc=20, label='pop', saving = F, plotname="observed, clusters (healthy ctrl)")

# counts <- cbind(counts_ctrl_healthy[[1]], counts_stim_healthy[[1]])
# tsne_batches <- PlotTsne(meta=observed_rnaseq_loBE[[2]][c(cellset_batch1, cellset_batch1),], data=log2(counts + 1), evf_type="discrete", n_pc=20, label='batch', saving = F, plotname="observed, batches (healthy )")
# tsne_clusters <- PlotTsne(meta=observed_rnaseq_loBE[[2]][c(cellset_batch1, cellset_batch1),], data=log2(counts + 1), evf_type="discrete", n_pc=20, label='pop', saving = F, plotname="observed, clusters (healthy ctrl)")

pdf(file=sprintf("%s/tsne.pdf", datapath))
print(tsne_batches[[2]])
print(tsne_clusters[[2]])
dev.off()
