import json
import torch
from espnet.bin.asr_train_x import get_parser 
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from collections import OrderedDict 
from espn_iter import get_iter
from espnet.asr.pytorch_backend.asr_init import load_trained_model
import itertools

import sys


onehot=False # toogle to switch between onehot and emb
train_json=sys.argv[1]
valid_json=sys.argv[2]
eval_json=sys.argv[3]
out=sys.argv[4]

# dump data
#train_json='data/json_data/mix200_sample.json'
#valid_json='data/json_data/dev20_sample.json'
#eval_json='data/json_data/dev20_sample.json'
#out='aux_emb'


# config
conf='/mnt/lustre/sjtu/home/jqg01/asr/e2e/cs/egs/codeswitching/asr/conf_aux_x/train_alldata_aux_emb_onehot_cov_ep1.yaml'
#train_json='/mnt/lustre/sjtu/users/yzl23/work_dir/asr/is20_codeswitching/espnet/egs/codeswitching/asr/data/json_data/mix200/data.json'
#valid_json='/mnt/lustre/sjtu/users/yzl23/work_dir/asr/is20_codeswitching/espnet/egs/codeswitching/asr/data/json_data/dev_mix20/data.json'
aux_model_path='/mnt/lustre/sjtu/users/mkh96/wordspace/asr/codeswitch/exp/phone_classifier/transformer_layer12_lsm0.0_ep100/results/snapshot.ep.100'
args, _ = get_parser(required=False).parse_known_args('--config {}  --train-json {} --valid-json {} --ngpu 0 '.format(conf, train_json, valid_json))
args2,_ = get_parser(required=False).parse_known_args('--config {}  --train-json {} --valid-json {} --ngpu 0 '.format(conf, train_json, eval_json))


# iter
tr_iter, dev_iter = get_iter(args)
_ , eval_iter = get_iter(args2)
tr_iter = tr_iter['main']
dev_iter = dev_iter['main']
eval_iter = eval_iter['main']


aux_model, aux_args = load_trained_model(aux_model_path)
aux_model.eval()

res=OrderedDict()

cnt=1
for batch in itertools.chain(tr_iter, dev_iter, eval_iter):
    xs_pad, ilens, ys_pad, uttid_list, train = batch
    xs_pad = xs_pad[:, :max(ilens)]  # for data parallel
    src_mask = (~make_pad_mask(ilens.tolist())).to(xs_pad.device).unsqueeze(-2)
    with torch.no_grad():
        emb, masks = aux_model.encoder(xs_pad, src_mask)
        masks = masks.squeeze(1)
        #print(masks, 'masks')
        #print(emb.size(), masks.size())
        if onehot:
            emb = aux_model.lid_lo(emb)
            batch_maxi = []
            for b in range(len(emb)):
                b_maxi = []
                list_maxi = emb[b].argmax(1)
                for i in range(len(masks[b])):
                    if masks[b][i] :
                        b_maxi.append(list_maxi[i].item())
                batch_maxi.append(b_maxi)


    for i in range(len(uttid_list)):
        uttid = uttid_list[i]
        emb_np = emb[i].data.cpu().numpy()
        if onehot:
            assert len(batch_maxi[i]) == len(str(batch_maxi[i]).replace(",","")[1:-1].split())
            res[uttid] = str(batch_maxi[i]).replace(",","")[1:-1]
        else:
            res[uttid]= emb_np #, file=tr_ali_aux2_drop_out)
        #print(uttid, res[uttid], emb_np)
    cnt += len(batch)
    print('cnt: {}'.format(str(cnt)), flush=True)

if onehot:
    aux_ali_out =  open(out_ali, 'w')
    for k in res:
        print(k ," ", res[k], file=aux_ali_out)
    aux_out.close()
else:
    from kaldiio import WriteHelper
    with WriteHelper('ark,scp:{}.ark,{}.scp'.format(out, out)) as writer:
        for k in res:
            writer(k, res[k])

