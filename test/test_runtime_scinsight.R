rm(list = ls())
gc()
library(scINSIGHT)
library(Seurat)
setwd("/localscratch/ziqi/scDisInFact/test/")


################################################################################
#
# Symsim datasets
#
################################################################################
dataset <- "imputation_10000_500_0.2_20_2"
data_dir <- paste0("../data/simulated_new/", dataset,"/")
result_dir <- "./simulated/runtime/"

times.total <- c()

for(stepsize in c(1,2,4,5,10)){
  print(stepsize)
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
  
  # sub-sampling
  GxC1 <- GxC1[,seq(1, dim(GxC1)[2], stepsize)]
  GxC2 <- GxC2[,seq(1, dim(GxC2)[2], stepsize)]
  GxC3 <- GxC3[,seq(1, dim(GxC3)[2], stepsize)]
  GxC4 <- GxC4[,seq(1, dim(GxC4)[2], stepsize)]
  GxC5 <- GxC5[,seq(1, dim(GxC5)[2], stepsize)]
  GxC6 <- GxC6[,seq(1, dim(GxC6)[2], stepsize)]
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
  
  start.time <- Sys.time()
  # Create an scINSIGHT object
  sim.scobj <- create_scINSIGHT(norm.data = sim.counts, condition = sim.condition)
  # K_j default is 2
  # conduct hyper-parameter search
  # sim.scobj <- run_scINSIGHT(sim.scobj, K = seq(5,15,2), K_j = 2, out.dir = './scinsight_intermid/', num.cores = 4)
  # no hyper-parameter search
  # sim.scobj <- run_scINSIGHT(sim.scobj, K = 13, K_j = 2, LDA = 0.01, out.dir = './scinsight_intermid/', num.cores = 4)
  # LDA has to be searched
  sim.scobj <- run_scINSIGHT(sim.scobj, K = 13, K_j = 2, out.dir = './scinsight_intermid/', num.cores = 4)
  end.time <- Sys.time()
  time.total <- end.time - start.time
  print(time.total)
  # append
  times.total <- c(times.total, time.total)
}

num.cells <- c(floor(10000), floor(10000/2), floor(10000/4), floor(10000/5), floor(10000/10))
runtime <- data.frame(num.cells, times.total)
write.table(runtime, file = paste0(result_dir, "runtime_scinsight_nosearch.csv"), sep = ",", quote = FALSE)
