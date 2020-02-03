ESPv3_ROOT=/mnt/lustre/sjtu/users/yzl23/work_dir/espnet_exp/espnet_v0.3.0
MAIN_ROOT=$PWD/../../..
KALDI_ROOT=${ESPv3_ROOT}/tools/kaldi

[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sctk/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${ESPv3_ROOT}/tools/chainer_ctc/ext/warp-ctc/build
if [ -e $ESPv3_ROOT/tools/venv/etc/profile.d/conda.sh ]; then
    source $ESPv3_ROOT/tools/venv/etc/profile.d/conda.sh && conda deactivate && conda activate
else
    source $ESPv3_ROOT/tools/venv/bin/activate
fi
export PATH=$MAIN_ROOT/utils:$MAIN_ROOT/espnet/bin:$PATH

export OMP_NUM_THREADS=1

# NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8

# additional lib needed
export LD_LIBRARY_PATH=/mnt/lustre/cm/shared/global/src/dev/gcc/7.3.0/lib64:$LD_LIBRARY_PATH

# PYTHONPATH
export PYTHONPATH=/mnt/lustre/sjtu/users/yzl23/work_dir/asr/is20_codeswitching/espnet:$PYTHONPATH



