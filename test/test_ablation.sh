n_diff_genes=100
sigma=0.2
diff=8
ncells_total=10000
ngenes=500

dataset=1condition_${ncells_total}_${ngenes}_${sigma}_${n_diff_genes}_${diff}
result_dir=ablation_test/${dataset}/
mkdir -p ${result_dir}

# only classifier
reg_mmd_comm=1e-3
reg_mmd_diff=1e-3
reg_gl=1
reg_tc=0.5
reg_class=1
reg_kl=1e-5
reg_contr=0.0
nohup python -u ablation_test.py ${dataset} ${reg_mmd_comm} ${reg_mmd_diff} ${reg_gl} ${reg_tc} ${reg_class} ${reg_kl} ${reg_contr} > ${result_dir}ablation_${reg_mmd_comm}_${reg_mmd_diff}_${reg_gl}_${reg_tc}_${reg_class}_${reg_kl}_${reg_contr}.out & 

# both classifier and contrastive loss
reg_mmd_comm=1e-3
reg_mmd_diff=1e-3
reg_gl=1
reg_tc=0.5
reg_class=1
reg_kl=1e-5
reg_contr=0.01
nohup python -u ablation_test.py ${dataset} ${reg_mmd_comm} ${reg_mmd_diff} ${reg_gl} ${reg_tc} ${reg_class} ${reg_kl} ${reg_contr} > ${result_dir}ablation_${reg_mmd_comm}_${reg_mmd_diff}_${reg_gl}_${reg_tc}_${reg_class}_${reg_kl}_${reg_contr}.out & 

# only contrastive loss
reg_mmd_comm=1e-3
reg_mmd_diff=1e-3
reg_gl=1
reg_tc=0.5
reg_class=0
reg_kl=1e-5
reg_contr=0.1
nohup python -u ablation_test.py ${dataset} ${reg_mmd_comm} ${reg_mmd_diff} ${reg_gl} ${reg_tc} ${reg_class} ${reg_kl} ${reg_contr} > ${result_dir}ablation_${reg_mmd_comm}_${reg_mmd_diff}_${reg_gl}_${reg_tc}_${reg_class}_${reg_kl}_${reg_contr}.out & 