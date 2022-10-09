## scDisInFact

### Description
scDisInFact is a single-cell data integration and condition effect prediction framework. Given a multi-batches multi-conditions scRNA-seq dataset (see figure below), scDisInFact is able to 
* Disentangling the shared-bio factors (condition-irrelevant) and unshared-bio factors (condition-related), and remove technical batch effect.
* Detect condition-associated key genes for each condition type.
* Predict the condition effect on gene expression data (Perturbation prediction) and remove the batch effect in gene expression data.

<img src = "figures/figure1.png" width = 900ptx>

scDisInFact is designed using a conditional variational autoencoder framework. See figure below for the network structure of scDisInFact:

<img src = "figures/figure2.png" width = 900ptx>

### Dependency
```
    python >= 3.7
    pytorch >= 1.11.0
    sklearn >= 0.22.2.post1
    numpy >= 1.19.5
    pandas >= 1.4.0
    scipy >- 1.7.3
    matplotlib >= 3.5.2
    umap >= 0.5.2
    adjustText >= 0.7.3 (optional)
```

### Directory
* `src` stores the source code of scDisInFact.
* `test` stores the testing script of scDisInFact.
* `data` stores the testing data of scDisInFact.     

### Data
The test dataset in the manuscript is available upon request.

### Installation and usage
No installation is needed. Please check [demo.iipynb](https://github.com/ZhangLabGT/scDisInFact/blob/main/demo.ipynb) for the usage of scDisInFact.

### Contact
* Ziqi Zhang: ziqi.zhang@gatech.edu
* Xiuwei Zhang: xiuwei.zhang@gatech.edu

### Cite