# library("devtools")
# devtools::install_github('YosefLab/SymSim')
library("SymSim")

# Simulate multiple discrete populations
# When there are multiple populations, users need to provide a tree. A tree with five leaves (five populations) can be generated as follows:
phyla1 <- Phyla3()

ngenes <- 500
ncells_total <- 10000
min_popsize <- 1000
Sigma <- 0.4
nbatch <- 6
diff <- 2
n_diff_genes <- 20

# The true counts of the five populations can be simulated:
true_counts_res <- SimulateTrueCounts(ncells_total=ncells_total, min_popsize=min_popsize, i_minpop=2, ngenes=ngenes, nevf=10, evf_type="discrete", n_de_evf=9, vary="s", Sigma=Sigma, phyla=phyla1, randseed=0)
true_counts_res_dis <- true_counts_res
tsne_true_counts <- PlotTsne(meta=true_counts_res[[3]], data=log2(true_counts_res[[1]]+1), evf_type="discrete", n_pc=20, label='pop', saving = F, plotname="discrete populations (true counts)")
tsne_true_counts[[2]]

# write.table(true_counts_res[[1]], file = "true_counts.csv", quote = FALSE, sep = ",", row.names = FALSE, col.names = FALSE)
# write.table(true_counts_res[[3]]["pop"], file = "anno.csv", quote = FALSE, row.names = FALSE, col.names = FALSE)

# transform to observed counts, using UMI count
# Each genes needs to be assigned a gene length. We sample lengths from human transcript lengths.
data(gene_len_pool)
gene_len <- sample(gene_len_pool, ngenes, replace = FALSE)
observed_counts <- True2ObservedCounts(true_counts=true_counts_res[[1]], meta_cell=true_counts_res[[3]], protocol="UMI", alpha_mean=0.05, alpha_sd=0.02, gene_len=gene_len, depth_mean=5e4, depth_sd=3e3)
tsne_UMI_counts <- PlotTsne(meta=observed_counts[[2]], data=log2(observed_counts[[1]]+1), evf_type="discrete", n_pc=20, label='pop', saving = F, plotname="observed counts UMI")
tsne_UMI_counts[[2]]

# We can divide the data we simulated using the previous steps into multiple batches and add batch effects to each batch. 
observed_rnaseq_loBE <- DivideBatches(observed_counts_res = observed_counts, nbatch = nbatch, batch_effect_size = 1)
tsne_batches <- PlotTsne(meta=observed_rnaseq_loBE[[2]], data=log2(observed_rnaseq_loBE[[1]]+1), evf_type="discrete", n_pc=20, label='batch', saving = F, plotname="observed counts in batches")
tsne_batches[[2]]

cellset_batch1 <- which(observed_rnaseq_loBE[[2]]$batch==1)
cellset_batch2 <- which(observed_rnaseq_loBE[[2]]$batch==2)
cellset_batch3 <- which(observed_rnaseq_loBE[[2]]$batch==3)
cellset_batch4 <- which(observed_rnaseq_loBE[[2]]$batch==4)
cellset_batch5 <- which(observed_rnaseq_loBE[[2]]$batch==5)
cellset_batch6 <- which(observed_rnaseq_loBE[[2]]$batch==6)

counts_batch1 <- observed_rnaseq_loBE[[1]][, cellset_batch1]
counts_batch2 <- observed_rnaseq_loBE[[1]][, cellset_batch2]
counts_batch3 <- observed_rnaseq_loBE[[1]][, cellset_batch3]
counts_batch4 <- observed_rnaseq_loBE[[1]][, cellset_batch4]
counts_batch5 <- observed_rnaseq_loBE[[1]][, cellset_batch5]
counts_batch6 <- observed_rnaseq_loBE[[1]][, cellset_batch6]

# add time progressing genes
# assume 40 genes changed, 20 genes growing, 20 genes reducing
counts_batch3[1:n_diff_genes,] <- counts_batch3[1:n_diff_genes,] + matrix(runif(n_diff_genes * dim(counts_batch3)[2], min = 0, max = diff), n_diff_genes, dim(counts_batch3)[2])
counts_batch4[1:n_diff_genes,] <- counts_batch4[1:n_diff_genes,] + matrix(runif(n_diff_genes * dim(counts_batch4)[2], min = 0, max = diff), n_diff_genes, dim(counts_batch4)[2])
counts_batch5[1:n_diff_genes,] <- counts_batch5[1:n_diff_genes,] + matrix(runif(n_diff_genes * dim(counts_batch5)[2], min = 0, max = 2*diff), n_diff_genes, dim(counts_batch5)[2])
counts_batch6[1:n_diff_genes,] <- counts_batch6[1:n_diff_genes,] + matrix(runif(n_diff_genes * dim(counts_batch6)[2], min = 0, max = 2*diff), n_diff_genes, dim(counts_batch6)[2])
counts_batch3[(n_diff_genes+1):(2*n_diff_genes),] <- counts_batch3[(n_diff_genes+1):(2*n_diff_genes),] + matrix(runif(n_diff_genes * dim(counts_batch3)[2], min = -0.5*diff, max = 0), n_diff_genes, dim(counts_batch3)[2])
counts_batch4[(n_diff_genes+1):(2*n_diff_genes),] <- counts_batch4[(n_diff_genes+1):(2*n_diff_genes),] + matrix(runif(n_diff_genes * dim(counts_batch4)[2], min = -0.5*diff, max = 0), n_diff_genes, dim(counts_batch4)[2])
counts_batch5[(n_diff_genes+1):(2*n_diff_genes),] <- counts_batch5[(n_diff_genes+1):(2*n_diff_genes),] + matrix(runif(n_diff_genes * dim(counts_batch5)[2], min = -diff, max = 0), n_diff_genes, dim(counts_batch5)[2])
counts_batch6[(n_diff_genes+1):(2*n_diff_genes),] <- counts_batch6[(n_diff_genes+1):(2*n_diff_genes),] + matrix(runif(n_diff_genes * dim(counts_batch6)[2], min = -diff, max = 0), n_diff_genes, dim(counts_batch6)[2])
# make sure all nonnegative
counts_batch3[(n_diff_genes+1):(2*n_diff_genes),] <- pmax(counts_batch3[(n_diff_genes+1):(2*n_diff_genes),], 0)
counts_batch4[(n_diff_genes+1):(2*n_diff_genes),] <- pmax(counts_batch4[(n_diff_genes+1):(2*n_diff_genes),], 0)
counts_batch5[(n_diff_genes+1):(2*n_diff_genes),] <- pmax(counts_batch5[(n_diff_genes+1):(2*n_diff_genes),], 0)
counts_batch6[(n_diff_genes+1):(2*n_diff_genes),] <- pmax(counts_batch6[(n_diff_genes+1):(2*n_diff_genes),], 0)

# second set of conditions: batch 1, 3, 5 and batch 2, 4, 6
# assume 20 genes growing and 20 genes reducing
counts_batch2[(2*n_diff_genes + 1):(3*n_diff_genes),] <- counts_batch2[(2*n_diff_genes + 1):(3*n_diff_genes),] + matrix(runif(n_diff_genes * dim(counts_batch2)[2], min = 0, max = diff), n_diff_genes, dim(counts_batch2)[2])
counts_batch4[(2*n_diff_genes + 1):(3*n_diff_genes),] <- counts_batch4[(2*n_diff_genes + 1):(3*n_diff_genes),] + matrix(runif(n_diff_genes * dim(counts_batch4)[2], min = 0, max = diff), n_diff_genes, dim(counts_batch4)[2])
counts_batch6[(2*n_diff_genes + 1):(3*n_diff_genes),] <- counts_batch6[(2*n_diff_genes + 1):(3*n_diff_genes),] + matrix(runif(n_diff_genes * dim(counts_batch6)[2], min = 0, max = diff), n_diff_genes, dim(counts_batch6)[2])
counts_batch2[(3*n_diff_genes + 1):(4*n_diff_genes),] <- counts_batch2[(3*n_diff_genes + 1):(4*n_diff_genes),] + matrix(runif(n_diff_genes * dim(counts_batch2)[2], min = -0.5*diff, max = 0), n_diff_genes, dim(counts_batch2)[2])
counts_batch4[(3*n_diff_genes + 1):(4*n_diff_genes),] <- counts_batch4[(3*n_diff_genes + 1):(4*n_diff_genes),] + matrix(runif(n_diff_genes * dim(counts_batch4)[2], min = -0.5*diff, max = 0), n_diff_genes, dim(counts_batch4)[2])
counts_batch6[(3*n_diff_genes + 1):(4*n_diff_genes),] <- counts_batch6[(3*n_diff_genes + 1):(4*n_diff_genes),] + matrix(runif(n_diff_genes * dim(counts_batch6)[2], min = -0.5*diff, max = 0), n_diff_genes, dim(counts_batch6)[2])
# make sure all nonnegative
counts_batch2[(3*n_diff_genes + 1):(4*n_diff_genes),] <- pmax(counts_batch2[(3*n_diff_genes + 1):(4*n_diff_genes),], 0)
counts_batch4[(3*n_diff_genes + 1):(4*n_diff_genes),] <- pmax(counts_batch4[(3*n_diff_genes + 1):(4*n_diff_genes),], 0)
counts_batch6[(3*n_diff_genes + 1):(4*n_diff_genes),] <- pmax(counts_batch6[(3*n_diff_genes + 1):(4*n_diff_genes),], 0)


datapath <- paste0("dataset_", ncells_total, "_", ngenes, "_", Sigma, "_", n_diff_genes, "_", diff)
system(sprintf("mkdir %s", datapath))
write.table(counts_batch1, sprintf("%s/GxC1.txt", datapath), quote=F, row.names = F, col.names = F, sep = "\t")
write.table(counts_batch2, sprintf("%s/GxC2.txt", datapath), quote=F, row.names = F, col.names = F, sep = "\t")
write.table(counts_batch3, sprintf("%s/GxC3.txt", datapath), quote=F, row.names = F, col.names = F, sep = "\t")
write.table(counts_batch4, sprintf("%s/GxC4.txt", datapath), quote=F, row.names = F, col.names = F, sep = "\t")
write.table(counts_batch5, sprintf("%s/GxC5.txt", datapath), quote=F, row.names = F, col.names = F, sep = "\t")
write.table(counts_batch6, sprintf("%s/GxC6.txt", datapath), quote=F, row.names = F, col.names = F, sep = "\t")

write.table(true_counts_res$cell_meta[cellset_batch1,1:2], sprintf("%s/cell_label1.txt", datapath), quote=F, row.names = F, col.names = T, sep = "\t")
write.table(true_counts_res$cell_meta[cellset_batch2,1:2], sprintf("%s/cell_label2.txt", datapath), quote=F, row.names = F, col.names = T, sep = "\t")
write.table(true_counts_res$cell_meta[cellset_batch3,1:2], sprintf("%s/cell_label3.txt", datapath), quote=F, row.names = F, col.names = T, sep = "\t")
write.table(true_counts_res$cell_meta[cellset_batch4,1:2], sprintf("%s/cell_label4.txt", datapath), quote=F, row.names = F, col.names = T, sep = "\t")
write.table(true_counts_res$cell_meta[cellset_batch5,1:2], sprintf("%s/cell_label5.txt", datapath), quote=F, row.names = F, col.names = T, sep = "\t")
write.table(true_counts_res$cell_meta[cellset_batch6,1:2], sprintf("%s/cell_label6.txt", datapath), quote=F, row.names = F, col.names = T, sep = "\t")

# write.table(true_counts_res$cell_meta[cellset_batch1,3], sprintf("%s/pseudotime1.txt", datapath), 
#             quote=F, row.names = F, col.names = F, sep = "\t")
# write.table(true_counts_res$cell_meta[cellset_batch2,3], sprintf("%s/pseudotime2.txt", datapath), 
#             quote=F, row.names = F, col.names = F, sep = "\t")
# write.table(true_counts_res$cell_meta[cellset_batch3,3], sprintf("%s/pseudotime3.txt", datapath), 
#             quote=F, row.names = F, col.names = F, sep = "\t")
# write.table(true_counts_res$cell_meta[cellset_batch4,3], sprintf("%s/pseudotime4.txt", datapath), 
#             quote=F, row.names = F, col.names = F, sep = "\t")
# write.table(true_counts_res$cell_meta[cellset_batch5,3], sprintf("%s/pseudotime5.txt", datapath), 
#             quote=F, row.names = F, col.names = F, sep = "\t")
# write.table(true_counts_res$cell_meta[cellset_batch6,3], sprintf("%s/pseudotime6.txt", datapath), 
#             quote=F, row.names = F, col.names = F, sep = "\t")

counts <- cbind(counts_batch1, counts_batch2, counts_batch3, counts_batch4, counts_batch5, counts_batch6)

# Plot batches and clusters
tsne_batches <- PlotTsne(meta=observed_rnaseq_loBE[[2]][c(cellset_batch1, cellset_batch2, cellset_batch3, cellset_batch4, cellset_batch5, cellset_batch6),], data=log2(counts + 1), evf_type="discrete", n_pc=20, label='batch', saving = F, plotname="observed counts  batches (adjusted)")
tsne_clusters <- PlotTsne(meta=observed_rnaseq_loBE[[2]][c(cellset_batch1, cellset_batch2, cellset_batch3, cellset_batch4, cellset_batch5, cellset_batch6),], data=log2(counts + 1), evf_type="discrete", n_pc=20, label='pop', saving = F, plotname="observed counts clusters (adjusted)")

pdf(file=sprintf("%s/tsne.pdf", datapath))
print(tsne_batches[[2]])
print(tsne_clusters[[2]])
dev.off()
