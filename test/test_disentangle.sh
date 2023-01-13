n_diff_genes=50
sigma=0.2
diff=8
ncells_total=10000
ngenes=500

dataset=1condition_${ncells_total}_${ngenes}_${sigma}_${n_diff_genes}_${diff}
result_dir=simulated/prediction/${dataset}/
mkdir -p ${result_dir}
nohup python -u test_embedding.py ${dataset} > ${result_dir}test_embedding.out & 