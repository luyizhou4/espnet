#!/bin/bash
#SBATCH -J sync5
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -c 3
#SBATCH -o logs/ddp.%j
#SBATCH -x gqxx-01-075,gqxx-01-014,gqxx-01-121,gqxx-01-122,gqxx-01-003,gqxx-01-072,gqxx-01-071
#SBATCH --mem=50G
#SBATCH --array=1-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

hostname
nvidia-smi
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

# ddp related
rank=$((SLURM_ARRAY_TASK_ID-1))
world_size=$SLURM_ARRAY_TASK_COUNT


# general configuration
backend=pytorch
stage=3        # start from 0 if you need to start from data preparation
stop_stage=100
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot
log=100

# others
accum_grad=1
n_iter_processes=2
mtlalpha=0.2

preprocess_config=conf/specaug.yaml 
train_config=conf/train_alldata.yaml
decode_config=conf/decode.yaml

# decoding parameter
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
n_average=5

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_json=/mnt/lustre/sjtu/users/yzl23/work_dir/asr/is20_codeswitching/espnet/egs/codeswitching/asr/data/json_data/cn480_lib960_mix200_m200syn5_discard/data.json
valid_json=/mnt/lustre/sjtu/users/yzl23/work_dir/asr/is20_codeswitching/espnet/egs/codeswitching/asr/data/json_data/dev_mix20/data.json

tag=ddp_all_data/data_aug/AS2S_base_jca${mtlalpha}_gpu${world_size}_sync5_discard1200_50
expdir=exp/${tag}
mkdir -p ${expdir}


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Network Training"
    # store config files
    if [ $rank -eq 0 ]; then 
        cp ${preprocess_config} ${expdir}/
        cp ${train_config} ${expdir}/
    fi
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train_ddp.py \
        --rank ${rank} \
        --world-size ${world_size} \
        --config ${train_config} \
        --preprocess-conf ${preprocess_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --debugmode ${debugmode} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --accum-grad ${accum_grad} \
        --report-interval-iters ${log} \
        --n-iter-processes ${n_iter_processes} \
        --mtlalpha ${mtlalpha} \
        --model-module "espnet.nets.pytorch_backend.e2e_asr_transformer_yzl23:E2E" \
        --train-json ${train_json} \
        --valid-json ${valid_json}
fi

if [ $rank -eq 0 ]; then

    decode_dir="dev_mix20_beam8"
    if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
        echo "stage 4: Decoding"
        nj=48
        mkdir -p ${expdir}/${decode_dir}
        cp ${decode_config} ${expdir}/${decode_dir}
        if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
        recog_model=model.last${n_average}.avg.best
        average_checkpoints.py --backend ${backend} \
                       --snapshots ${expdir}/results/snapshot.ep.* \
                       --out ${expdir}/results/${recog_model} \
                       --num ${n_average}
        fi
        pids=() # initialize pids
        (
            # split data
            dev_root=/mnt/lustre/sjtu/users/yzl23/work_dir/asr/is20_codeswitching/espnet/egs/codeswitching/asr/data/json_data/dev_mix20
            splitjson.py --parts ${nj} ${dev_root}/data.json

            #### use CPU for decoding
            ngpu=0

            slurm.pl JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
                asr_recog.py \
                --config ${decode_config} \
                --ngpu ${ngpu} \
                --backend ${backend} \
                --recog-json ${dev_root}/split${nj}utt/data.JOB.json \
                --result-label ${expdir}/${decode_dir}/data.JOB.json \
                --model ${expdir}/results/${recog_model}

        ) &
        pids+=($!) # store background pids
        i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
        [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
        echo "Finished"
    fi

    if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
        ref_label=/mnt/lustre/sjtu/users/yzl23/work_dir/asr/is20_codeswitching/espnet/egs/codeswitching/asr/data/json_data/dev_mix20/text
        vocab=/mnt/lustre/sjtu/users/yzl23/work_dir/asr/is20_codeswitching/espnet/egs/codeswitching/asr/data/json_data/vocab 
        ignored_tokens="0_1000_4005_4006_4007"
        # parse label and compute MER
        concatjson.py ${expdir}/${decode_dir}/data.*.json > ${expdir}/${decode_dir}/data.json
        python ./local_yzl23/utils/read_json_label.py ${expdir}/${decode_dir}/data.json ${expdir}/${decode_dir}/result.txt
        cat ${expdir}/${decode_dir}/result.txt | sort > ${expdir}/${decode_dir}/result.txt.sort 
        # prep hypothesis, here we remove unk and then discrete_label
        python ./local_yzl23/utils/parse_labels.py ${expdir}/${decode_dir}/result.txt.sort ${expdir}/${decode_dir}/result.word.txt ${vocab} ${ignored_tokens} 
        cat ${expdir}/${decode_dir}/result.word.txt | sed -e 's/\s\+/ /g' | sort | \
            awk '{printf "%s", $1; for (i = 2; i <= NF; i++) printf " %s", toupper($i); printf "\n"}' \
            > ${expdir}/${decode_dir}/hyp.word.final.txt 

        # prep reference
        python ./local_yzl23/utils/discrete_label.py ${ref_label} ${expdir}/${decode_dir}/ref.word.txt
        cat ${expdir}/${decode_dir}/ref.word.txt | sed -e 's/\s\+/ /g' | sort > ${expdir}/${decode_dir}/ref.word.final.txt
        
        # compute mer
        python ./local_yzl23/utils/compute-mer.py ${expdir}/${decode_dir}/ref.word.final.txt ${expdir}/${decode_dir}/hyp.word.final.txt 2>&1 | tee ${expdir}/${decode_dir}/score.result
    fi


    decode_dir="eval_mix20_beam8"
    if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
        echo "stage 6: Decoding"
        nj=48
        mkdir -p ${expdir}/${decode_dir}
        cp ${decode_config} ${expdir}/${decode_dir}
        #if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
        recog_model=model.last${n_average}.avg.best
        #average_checkpoints.py --backend ${backend} \
                       #--snapshots ${expdir}/results/snapshot.ep.* \
                       #--out ${expdir}/results/${recog_model} \
                       #--num ${n_average}
        #fi
        pids=() # initialize pids
        (
            # split data
            dev_root=/mnt/lustre/sjtu/users/yzl23/work_dir/asr/is20_codeswitching/espnet/egs/codeswitching/asr/data/json_data/eval_mix20
            splitjson.py --parts ${nj} ${dev_root}/data.json

            #### use CPU for decoding
            ngpu=0

            slurm.pl JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
                asr_recog.py \
                --config ${decode_config} \
                --ngpu ${ngpu} \
                --backend ${backend} \
                --recog-json ${dev_root}/split${nj}utt/data.JOB.json \
                --result-label ${expdir}/${decode_dir}/data.JOB.json \
                --model ${expdir}/results/${recog_model}

        ) &
        pids+=($!) # store background pids
        i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
        [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
        echo "Finished"
    fi

    if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
        ref_label=/mnt/lustre/sjtu/users/yzl23/work_dir/asr/is20_codeswitching/espnet/egs/codeswitching/asr/data/json_data/ground-truth.eval_mix20.text
        vocab=/mnt/lustre/sjtu/users/yzl23/work_dir/asr/is20_codeswitching/espnet/egs/codeswitching/asr/data/json_data/vocab 
        ignored_tokens="0_1000_4005_4006_4007"
        # parse label and compute MER
        concatjson.py ${expdir}/${decode_dir}/data.*.json > ${expdir}/${decode_dir}/data.json
        python ./local_yzl23/utils/read_json_label.py ${expdir}/${decode_dir}/data.json ${expdir}/${decode_dir}/result.txt
        cat ${expdir}/${decode_dir}/result.txt | sort > ${expdir}/${decode_dir}/result.txt.sort 
        # prep hypothesis, here we remove unk and then discrete_label
        python ./local_yzl23/utils/parse_labels.py ${expdir}/${decode_dir}/result.txt.sort ${expdir}/${decode_dir}/result.word.txt ${vocab} ${ignored_tokens} 
        cat ${expdir}/${decode_dir}/result.word.txt | sed -e 's/\s\+/ /g' | sort | \
            awk '{printf "%s", $1; for (i = 2; i <= NF; i++) printf " %s", toupper($i); printf "\n"}' \
            > ${expdir}/${decode_dir}/hyp.word.final.txt 

        # prep reference
        cat ${ref_label} | sed -e 's/\s\+/ /g' | sort > ${expdir}/${decode_dir}/ref.word.final.txt

        # compute mer
        python ./local_yzl23/utils/compute-mer.py ${expdir}/${decode_dir}/ref.word.final.txt ${expdir}/${decode_dir}/hyp.word.final.txt 2>&1 | tee ${expdir}/${decode_dir}/score.result
    fi
fi
