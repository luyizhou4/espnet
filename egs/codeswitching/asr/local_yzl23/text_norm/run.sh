#!/bin/bash

. ./cmd.sh
. ./path.sh

stage=5
stop_stage=5
export LC_ALL=en_US.UTF-8
# sentencepiece
export PATH=/mnt/lustre/sjtu/users/yzl23/.local/usr/local/bin/:${PATH}

# remove unrelated symbols
function process_text() {
    sed 's#\[[S|N|T|P]\]# #g;s#[@|~|,|.|!|?|(|)|:|"|*|-]# #g;s#？# #g;s#，# #g;s#、# #g;s#；# #g;s#。# #g;s#（# #g;s#）# #g;s#“# #g;s#”# #g;s#！# #g;s#　# #g;s#《# #g;s#》# #g;s#…# #g' $1 >> $2
}

# map number to chinese character &&  transform English character to upper case
function map_text() {
    awk 'BEGIN {
        f["0"] = "零"
        f["1"] = "一"
        f["2"] = "二"
        f["3"] = "三"
        f["4"] = "四"
        f["5"] = "五"
        f["6"] = "六"
        f["7"] = "七"
        f["8"] = "八"
        f["9"] = "九"
        }
        { split($0, chars, "");
            for (i = 1; i <= length(chars); i++) {
                c = chars[i]
                if (c in f) {
                    printf "%s", f[c];
                } else {
                    printf "%s", toupper(c);
                }
            }
            printf "\n"
        }' $1
}

# -----------
# config root_dir and output_root_dir
root=data/kaldi_data
oroot=data/json_data/text_transform
idir=${oroot}/itxt # store raw text
mkdir -p ${idir}

libri960=${idir}/libri960.txt
libri_dev=${idir}/libri_dev.txt
cn500_mix200=${idir}/cn500_mix200.txt # cn500h + mix200h text
dev_mix20=${idir}/dev_mix20.txt # dev_mix20

# normalize text
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage1: prepare normalized text into ${idir}"
    # librispeech text
    cp ${root}/lib960/text ${libri960}
    cp ${root}/lib_dev/text ${libri_dev}

    # text normalize: mix200, cn500, dev_mix20
    :>${cn500_mix200}
    process_text ${root}/mix200/text ${cn500_mix200}
    process_text ${root}/cn500/text ${cn500_mix200}
    :>${dev_mix20}
    process_text ${root}/dev_mix20/text ${dev_mix20}

    # map number [0-9] to chinese ,and make EN words upper case: cn500_mix200, dev_mix20
    # NOTE: utt_id is first removed and later pasted
    cut -d ' ' -f2- ${cn500_mix200} | map_text - | \
        paste -d ' ' <(awk '{print $1}' ${cn500_mix200}) - > ${cn500_mix200}.map
    mv ${cn500_mix200}.map ${cn500_mix200}

    cut -d ' ' -f2- ${dev_mix20} | map_text - | \
        paste -d ' ' <(awk '{print $1}' ${dev_mix20}) - > ${dev_mix20}.map
    mv ${dev_mix20}.map ${dev_mix20}
    
    echo "stage1 finish"
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage2: get word frequency and english text file for stage3 usage"
    # count chinese characters
    python local_yzl23/text_norm/extract_cn.py <(cut -d ' ' -f2- ${cn500_mix200}) > ${idir}/cn.freq

    # prepare english
    python local_yzl23/text_norm/extract_en.py <(cut -d ' ' -f2- ${cn500_mix200}) > ${idir}/en.txt
    cut -d ' ' -f2- ${libri960} >> ${idir}/en.txt
    # count english words
    awk '{ for (i = 1; i <= NF; i++) f[$i]++; } END { for (k in f) print k, f[k] }' ${idir}/en.txt |\
        sort -nrk2 > ${idir}/en.freq
fi

en_vocab_size=1000
bpe_dir=${idir}/bpe
mkdir -p ${bpe_dir}

odir=${oroot}/otxt
mkdir -p ${odir}

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Learning BPE with en_vocab_size=${en_vocab_size}, This may take a while..."
    spm_train --bos_id=-1 \
            --eos_id=-1 \
            --unk_id=0 \
            --input=${idir}/en.txt \
            --model_prefix=${bpe_dir}/bpe \
            --vocab_size=${en_vocab_size} \
            --character_coverage=1.0 \
            --model_type=bpe

    # -------- directly parse librispeech BPE id --------------------
    cp ${libri960} ${odir}/libri960.txt
    paste -d ' ' <(awk '{print $1}' ${libri960}) <(cut -d ' ' -f 2- ${libri960} | \
            spm_encode --model=${bpe_dir}/bpe.model --output_format=id) > ${odir}/libri960.id
    
    cp ${libri_dev} ${odir}/libri_dev.txt
    paste -d ' ' <(awk '{print $1}' ${libri_dev}) <(cut -d ' ' -f 2- ${libri_dev} | \
            spm_encode --model=${bpe_dir}/bpe.model --output_format=id) > ${odir}/libri_dev.id

    # -------- merge vocabulary --------------------
    # en & cn vocab
    awk '{if ($2 >= 25) print $1}' ${idir}/cn.freq > ${odir}/cn.vocab
    awk '{print $1}' ${bpe_dir}/bpe.vocab | tail -n +2 > ${odir}/en.vocab # ignore <unk> in bpe vocab
    # combine vocab
    cat ${odir}/en.vocab > ${odir}/vocab
    echo '<unk>' >> ${odir}/vocab
    cat ${odir}/cn.vocab >> ${odir}/vocab
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage4: get word frequency and english text file for stage3 usage"
    # convert text to id: set unk=1000, chinese character start from 1001
    python local_yzl23/text_norm/map_txt2id_yzl23.py \
        ${cn500_mix200}  ${odir}/cn.vocab ${en_vocab_size} ${bpe_dir}/bpe.model \
        > ${odir}/cn500_mix200.id

    cat ${cn500_mix200} | grep "mix200" > ${odir}/mix200.txt
    cat ${cn500_mix200} | grep "cn500" > ${odir}/cn500.txt
    cat ${odir}/cn500_mix200.id | grep "mix200" > ${odir}/mix200.id
    cat ${odir}/cn500_mix200.id | grep "cn500" > ${odir}/cn500.id

    python local_yzl23/text_norm/map_txt2id_yzl23.py \
        ${dev_mix20}  ${odir}/cn.vocab ${en_vocab_size} ${bpe_dir}/bpe.model \
        > ${odir}/dev_mix20.id
    cp ${dev_mix20} ${odir}/dev_mix20.txt
    echo "stage4 finish"
fi

# for lib_testclean、 lib_testother

# paste -d ' ' <(awk '{print $1}' text) <(cut -d ' ' -f 2- text | \
#             spm_encode --model=../text_transform/itxt/bpe/bpe.model --output_format=id) > ./label.id


# prepare mix200_sync1, mix200_sync5
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage5: prep mix200_sync1, mix200_sync5 text data"
    data_set="mix200_sync1 mix200_sync5"

    for subdir in ${data_set}; do
        subdir_txt=${idir}/${subdir}.txt
        # text normalize
        :>${subdir_txt}
        process_text ${root}/${subdir}/text ${subdir_txt}
        # map number [0-9] to chinese ,and make EN words upper case
        # NOTE: utt_id is first removed and later pasted
        cut -d ' ' -f2- ${subdir_txt} | map_text - | \
            paste -d ' ' <(awk '{print $1}' ${subdir_txt}) - > ${subdir_txt}.map
        mv ${subdir_txt}.map ${subdir_txt}

        # convert text to id: set unk=1000, chinese character start from 1001
        python local_yzl23/text_norm/map_txt2id_yzl23.py \
            ${subdir_txt}  ${odir}/cn.vocab ${en_vocab_size} ${bpe_dir}/bpe.model \
            > ${odir}/${subdir}.id
        cp ${subdir_txt} ${odir}/${subdir}.txt
    done 
    echo "stage5 finish"
fi

