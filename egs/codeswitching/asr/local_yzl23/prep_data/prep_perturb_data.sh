#!/bin/bash
#SBATCH --job-name=prep
#SBATCH --mem=80G
#SBATCH --cpus-per-task=1
#SBATCH --exclude=gqxx-01-039

. ./path.sh


stage=2
stop_stage=2

kaldi_data_root=data/kaldi_data
fbankdir=data/fbank
dumpdir=data/dump
tardir=data/json_data

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    mix200_dir=${kaldi_data_root}/mix200
    mix200sp_dir=${kaldi_data_root}/mix200sp
    mkdir -p ${mix200sp_dir}

    # utils/data/perturb_data_dir_speed_3way.sh --always-include-prefix true \
    #     ${mix200_dir} ${mix200sp_dir}

    nj=56
    for sub_dir in mix200sp; do
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
    # awk text and label.id from mix200 into mix200sp
    awk '{printf "sp0.9-%s\nsp1.0-%s\nsp1.1-%s\n", $0, $0, $0}' ${tardir}/mix200/label.id |\
        sort > ${tardir}/mix200sp/label.id
    awk '{printf "sp0.9-%s\nsp1.0-%s\nsp1.1-%s\n", $0, $0, $0}' ${tardir}/mix200/text |\
        sort > ${tardir}/mix200sp/text


    echo "stage 1: Feature Generation Finished."
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    mix200_dir=${kaldi_data_root}/cn500_tran480
    mix200sp_dir=${kaldi_data_root}/cn500_tran480sp
    mkdir -p ${mix200sp_dir}

    utils/data/perturb_data_dir_speed_3way.sh --always-include-prefix true \
        ${mix200_dir} ${mix200sp_dir}

    nj=56
    for sub_dir in cn500_tran480sp; do
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
        # cp ${kaldi_data_root}/${sub_dir}/text ${tardir}/${sub_dir}/text
        # cp ${kaldi_data_root}/${sub_dir}/text.ori ${tardir}/${sub_dir}/text.ori
    done
    # awk text and label.id from mix200 into mix200sp
    awk '{printf "sp0.9-%s\nsp1.0-%s\nsp1.1-%s\n", $0, $0, $0}' ${tardir}/cn500_tran480/label.id |\
        sort > ${tardir}/cn500_tran480sp/label.id
    awk '{printf "sp0.9-%s\nsp1.0-%s\nsp1.1-%s\n", $0, $0, $0}' ${tardir}/cn500_tran480/text |\
        sort > ${tardir}/cn500_tran480sp/text


    echo "stage 2: Feature Generation Finished."
fi

