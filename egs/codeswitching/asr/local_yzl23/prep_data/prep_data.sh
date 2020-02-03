#!/bin/bash
#SBATCH --job-name=prep
#SBATCH --mem=80G
#SBATCH --cpus-per-task=1
#SBATCH --exclude=gqxx-01-039

. ./cmd.sh
. ./path.sh

stage=1
stop_stage=1

# data_set="cn500 mix200 lib960 lib_dev dev_mix20 lib_testclean lib_testother eval_mix20"
# data_set="lib_dev"
data_set="mix200_sync5"

# make fbank feats
kaldi_data_root=data/kaldi_data
fbankdir=data/fbank
dumpdir=data/dump
tardir=data/json_data

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Feature Generation"
    nj=56
    for sub_dir in ${data_set}; do
        mkdir -p ${fbankdir}/${sub_dir}
        utils/validate_data_dir.sh --no-text --no-feats \
            ${kaldi_data_root}/${sub_dir}
        steps/make_fbank.sh --cmd slurm.pl --nj ${nj} --write_utt2num_frames true \
            ${kaldi_data_root}/${sub_dir} exp/make_fbank/${sub_dir} ${fbankdir}/${sub_dir}
        # compute per utterance cmvn
        compute-cmvn-stats --spk2utt=ark:${kaldi_data_root}/${sub_dir}/spk2utt \
            scp:${kaldi_data_root}/${sub_dir}/feats.scp \
            ark,scp:${kaldi_data_root}/${sub_dir}/cmvn_utt.ark,${kaldi_data_root}/${sub_dir}/cmvn_utt.scp
        
        mkdir -p ${dumpdir}/${sub_dir}
        dump_spk_yzl23.sh --cmd slurm.pl --nj ${nj} \
            ${kaldi_data_root}/${sub_dir}/feats.scp \
            ${kaldi_data_root}/${sub_dir}/cmvn_utt.scp \
            exp/dump_feats/${sub_dir} \
            ${dumpdir}/${sub_dir} \
            ${kaldi_data_root}/${sub_dir}/utt2spk
        
        # restore feats.scp in data_json dir
        mkdir -p ${tardir}/${sub_dir}
        cp ${dumpdir}/${sub_dir}/feats.scp ${tardir}/${sub_dir}
        cp ${dumpdir}/${sub_dir}/utt2num_frames ${tardir}/${sub_dir}
        cp ${kaldi_data_root}/${sub_dir}/utt2spk ${tardir}/${sub_dir}
        cp ${kaldi_data_root}/${sub_dir}/spk2utt ${tardir}/${sub_dir}
        cp ${kaldi_data_root}/${sub_dir}/text ${tardir}/${sub_dir}/text.ori

    done
    echo "stage 1: Feature Generation Finished."
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Text Normalization"    
    # run local_yzl23/text_norm/run.sh
    bash ./local_yzl23/text_norm/run.sh
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Prep eval data"
    sub_dir=eval_mix20
    nj=32
    mkdir -p ${fbankdir}/${sub_dir}
    utils/validate_data_dir.sh --no-text --no-feats \
        ${kaldi_data_root}/${sub_dir}
    steps/make_fbank.sh --cmd slurm.pl --nj ${nj} --write_utt2num_frames true \
        ${kaldi_data_root}/${sub_dir} exp/make_fbank/${sub_dir} ${fbankdir}/${sub_dir}
    # compute per utterance cmvn
    compute-cmvn-stats --spk2utt=ark:${kaldi_data_root}/${sub_dir}/spk2utt \
        scp:${kaldi_data_root}/${sub_dir}/feats.scp \
        ark,scp:${kaldi_data_root}/${sub_dir}/cmvn_utt.ark,${kaldi_data_root}/${sub_dir}/cmvn_utt.scp
    
    mkdir -p ${dumpdir}/${sub_dir}
    dump_spk_yzl23.sh --cmd slurm.pl --nj ${nj} \
        ${kaldi_data_root}/${sub_dir}/feats.scp \
        ${kaldi_data_root}/${sub_dir}/cmvn_utt.scp \
        exp/dump_feats/${sub_dir} \
        ${dumpdir}/${sub_dir} \
        ${kaldi_data_root}/${sub_dir}/utt2spk
    
    # restore feats.scp in data_json dir
    mkdir -p ${tardir}/${sub_dir}
    cp ${dumpdir}/${sub_dir}/feats.scp ${tardir}/${sub_dir}
    cp ${dumpdir}/${sub_dir}/utt2num_frames ${tardir}/${sub_dir}
    cp ${kaldi_data_root}/${sub_dir}/utt2spk ${tardir}/${sub_dir}
    cp ${kaldi_data_root}/${sub_dir}/spk2utt ${tardir}/${sub_dir}
    awk '{printf "%s FAKE_TRANSCRIPT\n",$1 }' ${tardir}/${sub_dir}/feats.scp \
        > ${tardir}/${sub_dir}/text
    awk '{printf "%s 0 0\n",$1 }' ${tardir}/${sub_dir}/feats.scp \
        > ${tardir}/${sub_dir}/label.id

    # generate json file for eval_mix20 set
    python ./local_yzl23/text_norm/generate_espnet_json_lid.py ${tardir}/${sub_dir} 80 4004 0_0 1000 0

    # record utt_number for json files    
    echo "stage 3 finish"
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Create Json files for training"
    python ./generate_espnet_json.py ./dev_mix20 80 4006 0_0 > ./dev_mix20/json_count
fi
