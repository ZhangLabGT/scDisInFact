reg_mmd_comm=1e-4
reg_mmd_diff=1e-4
reg_gl=1
reg_tc=0.5
reg_class=1
reg_kl=1e-6

nohup python -u test_sepsis_parameters.py ${reg_mmd_comm} ${reg_mmd_diff} ${reg_gl} ${reg_tc} ${reg_class} ${reg_kl}> test_sepsis_${reg_mmd_comm}_${reg_mmd_diff}_${reg_class}_${reg_gl}_${reg_tc}_${reg_kl}_8_4.out & 
# nohup python -u test_sepsis.py ${reg_mmd_comm} ${reg_mmd_diff} ${reg_gl} ${reg_tc} ${reg_class} ${reg_kl}> sepsis_raw/secondary/batch_batches/test_sepsis_${reg_mmd_comm}_${reg_mmd_diff}_${reg_class}_${reg_gl}_${reg_tc}_${reg_kl}_8_4.out & 
# nohup python -u test_sepsis.py ${reg_mmd_comm} ${reg_mmd_diff} ${reg_gl} ${reg_tc} ${reg_class} ${reg_kl}> sepsis_raw/primary/patient_batches/test_sepsis_${reg_mmd_comm}_${reg_mmd_diff}_${reg_class}_${reg_gl}_${reg_tc}_${reg_kl}_8_4.out & 
# nohup python -u test_sepsis.py ${reg_mmd_comm} ${reg_mmd_diff} ${reg_gl} ${reg_tc} ${reg_class} ${reg_kl}> sepsis_raw/primary/batch_batches/test_sepsis_${reg_mmd_comm}_${reg_mmd_diff}_${reg_class}_${reg_gl}_${reg_tc}_${reg_kl}_8_4.out & 
