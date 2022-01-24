# Documentation of AntennaVAE Project

## Jupyter files:

- **AE_model_pancreas_improve.ipynb: The current used file for dataset pancreas**

## Output Folder

### 1. output_diff_latent_dimension: Store experiment results of using different NN structure

- 32dim: (512, 128, 64, 32) layers
- 16dim_simple: (256, 64, 16) layers
- 8dim: (512, 128, 32, 8) layers
- 8dim_simple: (128, 32, 8) layers
- 4dim: (256, 64, 16, 4) layers

### 2. sample_to_same_cell_num: Store experiment results of making two batches have the same cell number by sampling

- 4_and_6: batch 4(1900+ cells) and batch 6(3600+ cells) 
- 7_and_6: batch 7(1300+ cells) and batch 6(3600+ cells) 
