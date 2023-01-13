reg_mmd_comm=1e-3
reg_mmd_diff=1e-3
reg_gl=1
reg_tc=0.5
reg_class=1
# with lr=5e-4, batchsize=64, reg_kl<1-5, larger than 180 epochs would result in nan
reg_kl=1e-5
reg_contr=0.01
nepochs=200
lr=5e-4
batch_size=64

nohup python -u test_integrated_covid.py ${reg_mmd_comm} ${reg_mmd_diff} ${reg_gl} ${reg_tc} ${reg_class} ${reg_kl} ${nepochs} ${lr} ${batch_size}> covid_hyperparams/test_covid_${reg_mmd_comm}_${reg_mmd_diff}_${reg_class}_${reg_gl}_${reg_tc}_${reg_kl}_${reg_contr}_${nepochs}_${lr}_${batch_size}_8_4_4.out & 
