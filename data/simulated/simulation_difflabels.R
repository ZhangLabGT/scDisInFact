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
phyla1 <- read.tree(text="(((A:1,B:1,F:1.5):1,(C:0.5,D:0.5,G:1.5):1.5,(H:0.5,I:0.5,J:1.5):2.0):1,((K:1,L:1,M:1.5):2.5,(N:0.5,O:0.5,P:1.5):3.0):2,E:3);")

ngenes <- 500
min_popsize <- 100
Sigma <- 0.4

# Config for 3 condition labels, 1 conditions type, 2 batches. 
ncells_total <- 10000
nbatch <- 2
epsilon <- 8
n_diff_genes <- 20
num_condlabels  <- 5
num_conds  <- 1

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

cellset_b1 <- which(observed_rnaseq_loBE[[2]]$batch==1)
cellset_b2 <- which(observed_rnaseq_loBE[[2]]$batch==2)

# true counts
counts_b1 <- true_counts_res[[1]][, cellset_b1]
counts_b2 <- true_counts_res[[1]][, cellset_b2]
counts_true <- list(counts_b1, counts_b2)

# cond1
counts_c1b1 <- observed_rnaseq_loBE[[1]][, cellset_b1]
counts_c1b2 <- observed_rnaseq_loBE[[1]][, cellset_b2]
counts_c1 <- list(counts_c1b1, counts_c1b2)

# cond1 + cond2
epsilon_per_cond<-epsilon/(num_condlabels-1)
interval<-epsilon_per_cond
counts <- list("c1" = counts_c1)
for(condlabel in seq(2, num_condlabels)){
  count <- lapply(counts_c1, function(x){
                    # use uniform distribution
                    x[1:n_diff_genes,] <- x[1:n_diff_genes,] + matrix(runif(n_diff_genes * dim(x)[2], min = -interval, max = 0) + (condlabel-1)*epsilon_per_cond, n_diff_genes, dim(x)[2])
                    return(x)})
  counts[[paste0("c", condlabel)]] <- count
}

print(paste0("simulated_robustness_", ncells_total, "_", nbatch, "_", num_condlabels, "_", num_conds, "_", epsilon, "_", n_diff_genes))
datapath <- paste0("simulated_robustness_", ncells_total, "_", nbatch, "_", num_condlabels, "_", num_conds, "_", epsilon, "_", n_diff_genes)
system(sprintf("mkdir -p %s", datapath))

write.table(counts_true[[1]], sprintf("%s/GxB1_true.txt", datapath), quote=F, row.names = F, col.names = F, sep = "\t")
write.table(counts_true[[2]], sprintf("%s/GxB2_true.txt", datapath), quote=F, row.names = F, col.names = F, sep = "\t")

write.table(true_counts_res$cell_meta[cellset_b1,1:2], sprintf("%s/cell_label_b1.txt", datapath), quote=F, row.names = F, col.names = T, sep = "\t")
write.table(true_counts_res$cell_meta[cellset_b2,1:2], sprintf("%s/cell_label_b2.txt", datapath), quote=F, row.names = F, col.names = T, sep = "\t")

for(condlabel in names(counts)){
  write.table(counts[[condlabel]][[1]], paste0(datapath, "/GxB1_", condlabel, ".txt"), quote=F, row.names = F, col.names = F, sep = "\t")
  write.table(counts[[condlabel]][[2]], paste0(datapath, "/GxB2_", condlabel, ".txt"), quote=F, row.names = F, col.names = F, sep = "\t")
}