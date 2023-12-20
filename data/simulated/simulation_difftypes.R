rm(list = ls())
gc()
# library("devtools")
# devtools::install_github('YosefLab/SymSim')
library("SymSim")
setwd("/localscratch/ziqi/scDisInFact/data/simulated/")

#-------------------------------------------------------------------------------
#
# Simulation parameter setting 
#
#-------------------------------------------------------------------------------
# Simulate multiple discrete populations
phyla1 <- read.tree(text="(((A:1,B:1,F:1.5):1,(C:0.5,D:0.5,G:1.5):1.5,(H:0.5,I:0.5,J:1.5):2.0):1,((K:1,L:1,M:1.5):2.5,(N:0.5,O:0.5,P:1.5):3.0):2,E:3);")

ngenes <- 500
min_popsize <- 100
Sigma <- 0.4

# Config for 3 condition labels, 1 conditions type, 2 batches. 
ncells_total <- 10000
nbatch <- 2
epsilon <- 4
n_diff_genes <- 20
num_condlabels  <- 2
num_conds  <- 4

# simulation function that generate the true count (without condition and batch effect)
true_counts_res <- SimulateTrueCounts(ncells_total=ncells_total, min_popsize=min_popsize, i_minpop=2, ngenes=ngenes, nevf=10, evf_type="discrete", n_de_evf=9, vary="s", Sigma=Sigma, phyla=phyla1, randseed=0)
true_counts_res_dis <- true_counts_res
tsne_true_counts <- PlotTsne(meta=true_counts_res[[3]], data=log2(true_counts_res[[1]]+1), evf_type="discrete", n_pc=20, label='pop', saving = F, plotname="discrete populations (true counts)")
tsne_true_counts[[2]]

#-------------------------------------------------------------------------------
#
# Generate observed count with batch effect 
#
#-------------------------------------------------------------------------------
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

print(paste0("simulated_robustness_", ncells_total, "_", nbatch, "_", num_condlabels, "_", num_conds, "_", epsilon, "_", n_diff_genes))
datapath <- paste0("simulated_robustness_", ncells_total, "_", nbatch, "_", num_condlabels, "_", num_conds, "_", epsilon, "_", n_diff_genes)
system(sprintf("mkdir -p %s", datapath))

#-------------------------------------------------------------------------------
#
# Add condition effect, 2 condition types: 
# (ctrl + healthy), (ctrl + mild), (ctrl + severe), (stim + healthy), (stim + mild), (stim + severe) 
#
#-------------------------------------------------------------------------------
# set random seed
set.seed(1)

cellset_b1 <- which(observed_rnaseq_loBE[[2]]$batch==1)
cellset_b2 <- which(observed_rnaseq_loBE[[2]]$batch==2)

# true counts
counts_b1 <- true_counts_res[[1]][, cellset_b1]
counts_b2 <- true_counts_res[[1]][, cellset_b2]
counts_true <- list(counts_b1, counts_b2)

if(num_conds == 2){
  counts_c11b1 <- observed_rnaseq_loBE[[1]][, cellset_b1]
  counts_c11b2 <- observed_rnaseq_loBE[[1]][, cellset_b2]
  counts_c11 <- list(counts_c11b1, counts_c11b2)
  
  interval<-1
  counts_c21 <- lapply(counts_c11, function(x){
    x[1:n_diff_genes,] <- x[1:n_diff_genes,] + matrix(runif(n_diff_genes * dim(x)[2], min = -interval, max = 0) + epsilon, n_diff_genes, dim(x)[2])
    return(x)
  })
  
  counts_c12 <- lapply(counts_c11, function(x){
    x[(n_diff_genes+1):(2*n_diff_genes),] <- x[(n_diff_genes+1):(2*n_diff_genes),] + matrix(runif(n_diff_genes * dim(x)[2], min = -interval, max = 0) + epsilon, n_diff_genes, dim(x)[2])
    return(x)
  })
  
  counts_c22 <- lapply(counts_c12, function(x){
    x[1:n_diff_genes,] <- x[1:n_diff_genes,] + matrix(runif(n_diff_genes * dim(x)[2], min = -interval, max = 0) + epsilon, n_diff_genes, dim(x)[2])
    return(x)
  })
  
  write.table(counts_c11[[1]], paste0(datapath, "/GxB1_c11.txt"), quote=F, row.names = F, col.names = F, sep = "\t")
  write.table(counts_c11[[2]], paste0(datapath, "/GxB2_c11.txt"), quote=F, row.names = F, col.names = F, sep = "\t")
  write.table(counts_c12[[1]], paste0(datapath, "/GxB1_c12.txt"), quote=F, row.names = F, col.names = F, sep = "\t")
  write.table(counts_c12[[2]], paste0(datapath, "/GxB2_c12.txt"), quote=F, row.names = F, col.names = F, sep = "\t")  
  write.table(counts_c21[[1]], paste0(datapath, "/GxB1_c21.txt"), quote=F, row.names = F, col.names = F, sep = "\t")
  write.table(counts_c21[[2]], paste0(datapath, "/GxB2_c21.txt"), quote=F, row.names = F, col.names = F, sep = "\t")
  write.table(counts_c22[[1]], paste0(datapath, "/GxB1_c22.txt"), quote=F, row.names = F, col.names = F, sep = "\t")
  write.table(counts_c22[[2]], paste0(datapath, "/GxB2_c22.txt"), quote=F, row.names = F, col.names = F, sep = "\t")
}else if(num_conds == 3){
  counts_c111b1 <- observed_rnaseq_loBE[[1]][, cellset_b1]
  counts_c111b2 <- observed_rnaseq_loBE[[1]][, cellset_b2]
  counts_c111 <- list(counts_c111b1, counts_c111b2)
  
  interval<-1
  counts_c211 <- lapply(counts_c111, function(x){
    x[1:n_diff_genes,] <- x[1:n_diff_genes,] + matrix(runif(n_diff_genes * dim(x)[2], min = -interval, max = 0) + epsilon, n_diff_genes, dim(x)[2])
    return(x)
  })
  counts_c121 <- lapply(counts_c111, function(x){
    x[(n_diff_genes+1):(2*n_diff_genes),] <- x[(n_diff_genes+1):(2*n_diff_genes),] + matrix(runif(n_diff_genes * dim(x)[2], min = -interval, max = 0) + epsilon, n_diff_genes, dim(x)[2])
    return(x)
  })
  counts_c112 <- lapply(counts_c111, function(x){
    x[(2*n_diff_genes+1):(3*n_diff_genes),] <- x[(2*n_diff_genes+1):(3*n_diff_genes),] + matrix(runif(n_diff_genes * dim(x)[2], min = -interval, max = 0) + epsilon, n_diff_genes, dim(x)[2])
    return(x)
  })
  counts_c122 <- lapply(counts_c121, function(x){
    x[(2*n_diff_genes+1):(3*n_diff_genes),] <- x[(2*n_diff_genes+1):(3*n_diff_genes),] + matrix(runif(n_diff_genes * dim(x)[2], min = -interval, max = 0) + epsilon, n_diff_genes, dim(x)[2])
    return(x)
  })
  counts_c212 <- lapply(counts_c112, function(x){
    x[1:n_diff_genes,] <- x[1:n_diff_genes,] + matrix(runif(n_diff_genes * dim(x)[2], min = -interval, max = 0) + epsilon, n_diff_genes, dim(x)[2])
    return(x)
  })
  counts_c221 <- lapply(counts_c211, function(x){
    x[(n_diff_genes+1):(2*n_diff_genes),] <- x[(n_diff_genes+1):(2*n_diff_genes),] + matrix(runif(n_diff_genes * dim(x)[2], min = -interval, max = 0) + epsilon, n_diff_genes, dim(x)[2])
    return(x)
  })
  counts_c222 <- lapply(counts_c212, function(x){
    x[(n_diff_genes+1):(2*n_diff_genes),] <- x[(n_diff_genes+1):(2*n_diff_genes),] + matrix(runif(n_diff_genes * dim(x)[2], min = -interval, max = 0) + epsilon, n_diff_genes, dim(x)[2])
    return(x)
  })
  
  write.table(counts_c111[[1]], paste0(datapath, "/GxB1_c111.txt"), quote=F, row.names = F, col.names = F, sep = "\t")
  write.table(counts_c111[[2]], paste0(datapath, "/GxB2_c111.txt"), quote=F, row.names = F, col.names = F, sep = "\t")
  write.table(counts_c211[[1]], paste0(datapath, "/GxB1_c211.txt"), quote=F, row.names = F, col.names = F, sep = "\t")
  write.table(counts_c211[[2]], paste0(datapath, "/GxB2_c211.txt"), quote=F, row.names = F, col.names = F, sep = "\t")  
  write.table(counts_c121[[1]], paste0(datapath, "/GxB1_c121.txt"), quote=F, row.names = F, col.names = F, sep = "\t")
  write.table(counts_c121[[2]], paste0(datapath, "/GxB2_c121.txt"), quote=F, row.names = F, col.names = F, sep = "\t")
  write.table(counts_c112[[1]], paste0(datapath, "/GxB1_c112.txt"), quote=F, row.names = F, col.names = F, sep = "\t")
  write.table(counts_c112[[2]], paste0(datapath, "/GxB2_c112.txt"), quote=F, row.names = F, col.names = F, sep = "\t")
  write.table(counts_c122[[1]], paste0(datapath, "/GxB1_c122.txt"), quote=F, row.names = F, col.names = F, sep = "\t")
  write.table(counts_c122[[2]], paste0(datapath, "/GxB2_c122.txt"), quote=F, row.names = F, col.names = F, sep = "\t")
  write.table(counts_c212[[1]], paste0(datapath, "/GxB1_c212.txt"), quote=F, row.names = F, col.names = F, sep = "\t")
  write.table(counts_c212[[2]], paste0(datapath, "/GxB2_c212.txt"), quote=F, row.names = F, col.names = F, sep = "\t")
  write.table(counts_c221[[1]], paste0(datapath, "/GxB1_c221.txt"), quote=F, row.names = F, col.names = F, sep = "\t")
  write.table(counts_c221[[2]], paste0(datapath, "/GxB2_c221.txt"), quote=F, row.names = F, col.names = F, sep = "\t")
  write.table(counts_c222[[1]], paste0(datapath, "/GxB1_c222.txt"), quote=F, row.names = F, col.names = F, sep = "\t")
  write.table(counts_c222[[2]], paste0(datapath, "/GxB2_c222.txt"), quote=F, row.names = F, col.names = F, sep = "\t")
}else if(num_conds == 4){
  counts_c1111b1 <- observed_rnaseq_loBE[[1]][, cellset_b1]
  counts_c1111b2 <- observed_rnaseq_loBE[[1]][, cellset_b2]
  counts_c1111 <- list(counts_c1111b1, counts_c1111b2)
  
  interval<-1
  counts_c2111 <- lapply(counts_c1111, function(x){
    x[1:n_diff_genes,] <- x[1:n_diff_genes,] + matrix(runif(n_diff_genes * dim(x)[2], min = -interval, max = 0) + epsilon, n_diff_genes, dim(x)[2])
    return(x)
  })
  counts_c1211 <- lapply(counts_c1111, function(x){
    x[(n_diff_genes+1):(2*n_diff_genes),] <- x[(n_diff_genes+1):(2*n_diff_genes),] + matrix(runif(n_diff_genes * dim(x)[2], min = -interval, max = 0) + epsilon, n_diff_genes, dim(x)[2])
    return(x)
  })
  counts_c1121 <- lapply(counts_c1111, function(x){
    x[(2*n_diff_genes+1):(3*n_diff_genes),] <- x[(2*n_diff_genes+1):(3*n_diff_genes),] + matrix(runif(n_diff_genes * dim(x)[2], min = -interval, max = 0) + epsilon, n_diff_genes, dim(x)[2])
    return(x)
  })
  counts_c1112 <- lapply(counts_c1111, function(x){
    x[(3*n_diff_genes+1):(4*n_diff_genes),] <- x[(3*n_diff_genes+1):(4*n_diff_genes),] + matrix(runif(n_diff_genes * dim(x)[2], min = -interval, max = 0) + epsilon, n_diff_genes, dim(x)[2])
    return(x)
  })
  counts_c2112 <- lapply(counts_c1112, function(x){
    x[1:n_diff_genes,] <- x[1:n_diff_genes,] + matrix(runif(n_diff_genes * dim(x)[2], min = -interval, max = 0) + epsilon, n_diff_genes, dim(x)[2])
    return(x)
  })
  counts_c1212 <- lapply(counts_c1112, function(x){
    x[(n_diff_genes+1):(2*n_diff_genes),] <- x[(n_diff_genes+1):(2*n_diff_genes),] + matrix(runif(n_diff_genes * dim(x)[2], min = -interval, max = 0) + epsilon, n_diff_genes, dim(x)[2])
    return(x)
  })
  counts_c1122 <- lapply(counts_c1112, function(x){
    x[(2*n_diff_genes+1):(3*n_diff_genes),] <- x[(2*n_diff_genes+1):(3*n_diff_genes),] + matrix(runif(n_diff_genes * dim(x)[2], min = -interval, max = 0) + epsilon, n_diff_genes, dim(x)[2])
    return(x)
  })
  counts_c2121 <- lapply(counts_c1121, function(x){
    x[1:n_diff_genes,] <- x[1:n_diff_genes,] + matrix(runif(n_diff_genes * dim(x)[2], min = -interval, max = 0) + epsilon, n_diff_genes, dim(x)[2])
    return(x)
  })
  counts_c1221 <- lapply(counts_c1121, function(x){
    x[(n_diff_genes+1):(2*n_diff_genes),] <- x[(n_diff_genes+1):(2*n_diff_genes),] + matrix(runif(n_diff_genes * dim(x)[2], min = -interval, max = 0) + epsilon, n_diff_genes, dim(x)[2])
    return(x)
  })
  counts_c2211 <- lapply(counts_c1211, function(x){
    x[1:n_diff_genes,] <- x[1:n_diff_genes,] + matrix(runif(n_diff_genes * dim(x)[2], min = -interval, max = 0) + epsilon, n_diff_genes, dim(x)[2])
    return(x)
  })
  counts_c2122 <- lapply(counts_c1122, function(x){
    x[1:n_diff_genes,] <- x[1:n_diff_genes,] + matrix(runif(n_diff_genes * dim(x)[2], min = -interval, max = 0) + epsilon, n_diff_genes, dim(x)[2])
    return(x)
  })
  counts_c1222 <- lapply(counts_c1122, function(x){
    x[(n_diff_genes+1):(2*n_diff_genes),] <- x[(n_diff_genes+1):(2*n_diff_genes),] + matrix(runif(n_diff_genes * dim(x)[2], min = -interval, max = 0) + epsilon, n_diff_genes, dim(x)[2])
    return(x)
  })
  counts_c2221 <- lapply(counts_c2121, function(x){
    x[(n_diff_genes+1):(2*n_diff_genes),] <- x[(n_diff_genes+1):(2*n_diff_genes),] + matrix(runif(n_diff_genes * dim(x)[2], min = -interval, max = 0) + epsilon, n_diff_genes, dim(x)[2])
    return(x)
  })
  counts_c2212 <- lapply(counts_c2112, function(x){
    x[(n_diff_genes+1):(2*n_diff_genes),] <- x[(n_diff_genes+1):(2*n_diff_genes),] + matrix(runif(n_diff_genes * dim(x)[2], min = -interval, max = 0) + epsilon, n_diff_genes, dim(x)[2])
    return(x)
  })
  counts_c2222 <- lapply(counts_c2122, function(x){
    x[(n_diff_genes+1):(2*n_diff_genes),] <- x[(n_diff_genes+1):(2*n_diff_genes),] + matrix(runif(n_diff_genes * dim(x)[2], min = -interval, max = 0) + epsilon, n_diff_genes, dim(x)[2])
    return(x)
  })
  
  write.table(counts_c1111[[1]], paste0(datapath, "/GxB1_c1111.txt"), quote=F, row.names = F, col.names = F, sep = "\t")
  write.table(counts_c1111[[2]], paste0(datapath, "/GxB2_c1111.txt"), quote=F, row.names = F, col.names = F, sep = "\t")
  write.table(counts_c2111[[1]], paste0(datapath, "/GxB1_c2111.txt"), quote=F, row.names = F, col.names = F, sep = "\t")
  write.table(counts_c2111[[2]], paste0(datapath, "/GxB2_c2111.txt"), quote=F, row.names = F, col.names = F, sep = "\t")  
  write.table(counts_c1211[[1]], paste0(datapath, "/GxB1_c1211.txt"), quote=F, row.names = F, col.names = F, sep = "\t")
  write.table(counts_c1211[[2]], paste0(datapath, "/GxB2_c1211.txt"), quote=F, row.names = F, col.names = F, sep = "\t")
  write.table(counts_c1121[[1]], paste0(datapath, "/GxB1_c1121.txt"), quote=F, row.names = F, col.names = F, sep = "\t")
  write.table(counts_c1121[[2]], paste0(datapath, "/GxB2_c1121.txt"), quote=F, row.names = F, col.names = F, sep = "\t")
  write.table(counts_c1221[[1]], paste0(datapath, "/GxB1_c1221.txt"), quote=F, row.names = F, col.names = F, sep = "\t")
  write.table(counts_c1221[[2]], paste0(datapath, "/GxB2_c1221.txt"), quote=F, row.names = F, col.names = F, sep = "\t")
  write.table(counts_c2121[[1]], paste0(datapath, "/GxB1_c2121.txt"), quote=F, row.names = F, col.names = F, sep = "\t")
  write.table(counts_c2121[[2]], paste0(datapath, "/GxB2_c2121.txt"), quote=F, row.names = F, col.names = F, sep = "\t")
  write.table(counts_c2211[[1]], paste0(datapath, "/GxB1_c2211.txt"), quote=F, row.names = F, col.names = F, sep = "\t")
  write.table(counts_c2211[[2]], paste0(datapath, "/GxB2_c2211.txt"), quote=F, row.names = F, col.names = F, sep = "\t")
  write.table(counts_c2221[[1]], paste0(datapath, "/GxB1_c2221.txt"), quote=F, row.names = F, col.names = F, sep = "\t")
  write.table(counts_c2221[[2]], paste0(datapath, "/GxB2_c2221.txt"), quote=F, row.names = F, col.names = F, sep = "\t")
  write.table(counts_c1112[[1]], paste0(datapath, "/GxB1_c1112.txt"), quote=F, row.names = F, col.names = F, sep = "\t")
  write.table(counts_c1112[[2]], paste0(datapath, "/GxB2_c1112.txt"), quote=F, row.names = F, col.names = F, sep = "\t")
  write.table(counts_c2112[[1]], paste0(datapath, "/GxB1_c2112.txt"), quote=F, row.names = F, col.names = F, sep = "\t")
  write.table(counts_c2112[[2]], paste0(datapath, "/GxB2_c2112.txt"), quote=F, row.names = F, col.names = F, sep = "\t")  
  write.table(counts_c1212[[1]], paste0(datapath, "/GxB1_c1212.txt"), quote=F, row.names = F, col.names = F, sep = "\t")
  write.table(counts_c1212[[2]], paste0(datapath, "/GxB2_c1212.txt"), quote=F, row.names = F, col.names = F, sep = "\t")
  write.table(counts_c1122[[1]], paste0(datapath, "/GxB1_c1122.txt"), quote=F, row.names = F, col.names = F, sep = "\t")
  write.table(counts_c1122[[2]], paste0(datapath, "/GxB2_c1122.txt"), quote=F, row.names = F, col.names = F, sep = "\t")
  write.table(counts_c1222[[1]], paste0(datapath, "/GxB1_c1222.txt"), quote=F, row.names = F, col.names = F, sep = "\t")
  write.table(counts_c1222[[2]], paste0(datapath, "/GxB2_c1222.txt"), quote=F, row.names = F, col.names = F, sep = "\t")
  write.table(counts_c2122[[1]], paste0(datapath, "/GxB1_c2122.txt"), quote=F, row.names = F, col.names = F, sep = "\t")
  write.table(counts_c2122[[2]], paste0(datapath, "/GxB2_c2122.txt"), quote=F, row.names = F, col.names = F, sep = "\t")
  write.table(counts_c2212[[1]], paste0(datapath, "/GxB1_c2212.txt"), quote=F, row.names = F, col.names = F, sep = "\t")
  write.table(counts_c2212[[2]], paste0(datapath, "/GxB2_c2212.txt"), quote=F, row.names = F, col.names = F, sep = "\t")
  write.table(counts_c2222[[1]], paste0(datapath, "/GxB1_c2222.txt"), quote=F, row.names = F, col.names = F, sep = "\t")
  write.table(counts_c2222[[2]], paste0(datapath, "/GxB2_c2222.txt"), quote=F, row.names = F, col.names = F, sep = "\t")
}

write.table(counts_true[[1]], sprintf("%s/GxB1_true.txt", datapath), quote=F, row.names = F, col.names = F, sep = "\t")
write.table(counts_true[[2]], sprintf("%s/GxB2_true.txt", datapath), quote=F, row.names = F, col.names = F, sep = "\t")

write.table(true_counts_res$cell_meta[cellset_b1,1:2], sprintf("%s/cell_label_b1.txt", datapath), quote=F, row.names = F, col.names = T, sep = "\t")
write.table(true_counts_res$cell_meta[cellset_b2,1:2], sprintf("%s/cell_label_b2.txt", datapath), quote=F, row.names = F, col.names = T, sep = "\t")

