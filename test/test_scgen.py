# In[]
import scanpy as sc
# import scgen
import sys
import pandas as pd
import scanpy as sc
import numpy as np
import anndata as ad
import os

# In[]
simu_data_dir_path = '../data/simulated_new/'
save_path = './simulated/imputation_scGEN/'
if not os.path.exists(save_path):
    os.mkdir(save_path)

data_paths = []
data_files = []
data_dir = os.walk(simu_data_dir_path)
for root, dirs, files in data_dir:
    for name in dirs:
        if 'imputation' in name:
            data_paths.append(os.path.join(root, name))

for this_simu_dir_path in data_paths:
  count_data = []
  cell_type = []
  batch_id = []
  conditions = []
  for i in range(1, 7):
    if i in [1, 2]:
      condition_name = 'ctrl'
    if i in [3, 4]:
      condition_name = 'stim1'
    if i in [5, 6]:
      condition_name = 'stim2'
    df = sc.read_text(os.path.join(this_simu_dir_path, 'GxC{}_{}.txt'.format(i, condition_name)))
    ct = pd.read_table(os.path.join(this_simu_dir_path, 'cell_label{}.txt'.format(i)))
    conditions.append([condition_name] * ct.shape[0])
    # batch_id.append([np.ceil(i/2)] * ct.shape[0])
    batch_id.append([i] * ct.shape[0])
    cell_type.append(ct['pop'].values)
    count_data.append(df.X)

  obs = pd.DataFrame()
  obs['batch'] = [str(i) for i in np.concatenate(batch_id)]
  obs['condition'] = np.concatenate(conditions)

  #! Use Fake cell type in order to perturbate all cell types
  # obs['cell_type'] = ['fake'] * np.concatenate(cell_type).shape[0]
  obs['cell_type'] = [str(i) for i in np.concatenate(cell_type)]
  simulated_ad = ad.AnnData(np.concatenate(count_data, axis=1).T, obs=obs)

  scgen.SCGEN.setup_anndata(simulated_ad, batch_key="condition", labels_key="cell_type")

  model = scgen.SCGEN(simulated_ad)
  model.train(
      max_epochs=100,
      batch_size=32,
      early_stopping=True,
      early_stopping_patience=25
  )
  pre_all_ct_stim1 = None
  pre_all_ct_stim2 = None
  for i in np.unique(np.concatenate(cell_type)):
    pred_stim1, delta = model.predict(
      ctrl_key='ctrl',
      stim_key='stim1',
      celltype_to_predict=str(i)
    )
    pred_stim1.obs['condition'] = 'pred'
    pred_stim2, delta = model.predict(
      ctrl_key='ctrl',
      stim_key='stim2',
      celltype_to_predict=str(i)
    )
    pred_stim2.obs['condition'] = 'pred'

    if i == 1:
      pre_all_ct_stim1 = pred_stim1
      pre_all_ct_stim2 = pred_stim2
    else:
      pre_all_ct_stim1 = pre_all_ct_stim1.concatenate(pred_stim1)
      pre_all_ct_stim1 = pre_all_ct_stim1.concatenate(pred_stim2)

  name_prefix = this_simu_dir_path.split('imputation')[1]
  pre_all_ct_stim1.write(os.path.join(save_path, 'imputed_stim1{}.h5ad'.format(name_prefix)))
  pre_all_ct_stim2.write(os.path.join(save_path, 'imputed_stim2{}.h5ad'.format(name_prefix)))


# In[]
for this_simu_dir_path in data_paths:
    name_prefix = this_simu_dir_path.split('imputation')[1]
    pre_all_ct_stim1 = sc.read_h5ad(os.path.join(save_path, 'imputed_stim1{}.h5ad'.format(name_prefix)))
    pre_all_ct_stim2 = sc.read_h5ad(os.path.join(save_path, 'imputed_stim2{}.h5ad'.format(name_prefix)))
    X_impute1 = pre_all_ct_stim1.X
    X_impute2 = pre_all_ct_stim2.X

    for i in range(1, 7):
        if i in [1, 2]:
            condition_name = 'ctrl'
        if i in [3, 4]:
            condition_name = 'stim1'
        if i in [5, 6]:
            condition_name = 'stim2'
        df = sc.read_text(os.path.join(this_simu_dir_path, 'GxC{}_{}.txt'.format(i, condition_name)))
        print(df.shape)
    break

# %%
