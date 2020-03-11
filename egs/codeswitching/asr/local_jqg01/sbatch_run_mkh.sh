#!/bin/bash


mkdir -p logs
log=logs/sbatch_history

function do_and_log () {
    local_cmd=$@
    echo $local_cmd >> $log
    eval $local_cmd >> $log
    echo `date` >> $log
    echo "" >> $log
    echo "" >> $log
}


cmd='sbatch script/distributed_data_parallel/all_data/run_AS2S_jca0.2_ddp8_mix200_aux_x.sh aux_oracleali_cov_drop_0.3_0.3_0.3' 
do_and_log $cmd
echo $cmd submit

cmd='sbatch script/distributed_data_parallel/all_data/run_AS2S_jca0.2_ddp8_mix200_aux_x.sh aux_oracleali_cov_drop_0.1_0.2_0.2' 
do_and_log $cmd
echo $cmd submit

