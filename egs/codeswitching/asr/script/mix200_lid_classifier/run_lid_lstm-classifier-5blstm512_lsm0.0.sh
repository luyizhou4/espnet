#!/bin/bash
#SBATCH -J 5b512
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -c 3
#SBATCH -o logs/%j
#SBATCH -x gqxx-01-075,gqxx-01-014,gqxx-00-005,gqxx-01-121,gqxx-01-122,gqxx-01-076
#SBATCH --mem=50G


# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

hostname
nvidia-smi
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

# general configuration
backend=pytorch
stage=3        # start from 0 if you need to start from data preparation
stop_stage=3
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot
log=100

# others
accum_grad=8
n_iter_processes=2

preprocess_config=conf/specaug.yaml 
train_config=./conf/train_lid_classifier_rnn.yaml
decode_config=conf/decode.yaml

# yzl23
elayers=5
lsm_weight=0.0
epochs=30
dropout_rate=0.2
eunits=512

# decoding parameter
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
n_average=5

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_json=/mnt/lustre/sjtu/users/yzl23/work_dir/asr/is20_codeswitching/espnet/egs/codeswitching/asr/data/lang_chunk/hybrid_align_json/mix200_tr/data.json
valid_json=/mnt/lustre/sjtu/users/yzl23/work_dir/asr/is20_codeswitching/espnet/egs/codeswitching/asr/data/lang_chunk/hybrid_align_json/mix200_cv/data.json

tag=lid_classifier/${elayers}blstm${eunits}_lsm${lsm_weight}_ep${epochs}
expdir=exp/${tag}
mkdir -p ${expdir}


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Network Training"
    # store config files
    cp ${preprocess_config} ${expdir}/
    cp ${train_config} ${expdir}/

    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
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
        --report-interval-iters ${log} \
        --accum-grad ${accum_grad} \
        --n-iter-processes ${n_iter_processes} \
        --elayers ${elayers} \
        --eunits ${eunits} \
        --dropout-rate ${dropout_rate} \
        --lsm-weight ${lsm_weight} \
        --epochs ${epochs} \
        --train-json ${train_json} \
        --valid-json ${valid_json}
fi


dec_tag="greedy_decoding"
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Decoding"
    nj=48
    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
	recog_model=model.last${n_average}.avg.best
    average_checkpoints.py --backend ${backend} \
                   --snapshots ${expdir}/results/snapshot.ep.* \
                   --out ${expdir}/results/${recog_model} \
                   --num ${n_average}
    fi
    pids=() # initialize pids
    for rtask in ${dec_tag}; do
    (
        decode_dir=decode_${rtask}
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
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    ref_label=/mnt/lustre/sjtu/users/yzl23/work_dir/asr/is20_codeswitching/espnet/egs/codeswitching/asr/data/json_data/dev_mix20/text
    vocab=/mnt/lustre/sjtu/users/yzl23/work_dir/asr/is20_codeswitching/espnet/egs/codeswitching/asr/data/json_data/vocab 
    ignored_tokens="0_1000_4005_4006_4007"
    decode_dir=decode_${dec_tag}
    cp ${decode_config} ${expdir}/${decode_dir}
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
