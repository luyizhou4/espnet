#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#    Apache 2.0    (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder definition."""

import torch
import logging
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import Conv1dLinear
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import MultiLayeredConv1d
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import PositionwiseFeedForward
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling

# auxiliary model relevant
#from espnet.nets.pytorch_backend.e2e_lid_transformer import E2E as aux_model
from espnet.asr.pytorch_backend.asr_init import load_trained_model
import numpy as np


class AuxModel(torch.nn.Module):
    """Auxiliary Model load phone ali to build auxliary onehot embedings, which act as oracle ali.   
    """
    def __init__(self, ali_file, ali_sampled=False):
        super(AuxModel, self).__init__()
        self.dim=128
        self.ali_file = ali_file
        self.ali_sampled = ali_sampled
        self.load_ali()

    def forward(self, uttid_list, maxlen, device, covsample=True): # uttid_list (b,), masks (b,c,l)
        return_batch = [] # (b, t, dim)
        for uttid in uttid_list:
            missing = False
            if uttid in self.uttid2ali:
                ali = self.uttid2ali[uttid]
            else:
                missing = True
                ali = []
                logging.warning('utt {} is missing in phone ali'.format(uttid))
            l_list = []
            for l in ali:
                l_list.append(l)
            l_list = np.array(l_list, dtype='int')
            one_hot = np.zeros((maxlen, self.dim)) # (t, dim)
            if missing == False:
                one_hot[np.arange(l_list.size), l_list] = 1
            #print(one_hot.shape,'1')
            if covsample:
                if self.ali_sampled:
                    end = (((maxlen - 1) // 2 - 1) // 2)
                    one_hot = one_hot[:end]
                    #print(one_hot.shape,'2',"onehot = ", one_hot, "len(l_kist) = " ,len(l_list) ,"end = ", end, "maxlen = ", maxlen)
                    #print(uttid, str(list(l_list)).replace(",","")[1:-1])
                else:
                    one_hot = one_hot[:-2:2,:][:-2:2,:] # Convsampling \times 2
            return_batch.append(one_hot)
        return_batch = np.array(return_batch) #List(narray(t, dim)) -> narray(b, t, dim)
        
        return torch.FloatTensor(return_batch).to(device)


    def load_ali(self):
        uttid2ali = {}
        with open(self.ali_file, 'r') as fin:
            for line in fin:
                uid, ali = line.split()[0], line.split()[1:]
                uttid2ali[uid] = ali
        self.uttid2ali = uttid2ali

    def load_phntab(self):
        ptab = {}
        dim = 0
        with open(self.phntab_file, 'r') as phnin:
            for line in phnin:
                phn, pid = line.split()[0], line.split()[1]
                ptab[phn] = int(pid)
                dim += 1
        self.ptab = ptab
        self.dim = dim


class Encoder(torch.nn.Module):
    """Transformer encoder module.

    :param int idim: input dim
    :param int attention_dim: dimention of attention
    :param int attention_heads: the number of heads of multi head attention
    :param int linear_units: the number of units of position-wise feed forward
    :param int num_blocks: the number of decoder blocks
    :param float dropout_rate: dropout rate
    :param float attention_dropout_rate: dropout rate in attention
    :param float positional_dropout_rate: dropout rate after adding positional encoding
    :param str or torch.nn.Module input_layer: input layer type
    :param class pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output
    if True, additional linear will be applied. i.e. x -> x + linear(concat(x, att(x)))
    if False, no additional linear will be applied. i.e. x -> x + att(x)
    :param str positionwise_layer_type: linear of conv1d
    :param int positionwise_conv_kernel_size: kernel size of positionwise conv1d layer
    :param int padding_idx: padding_idx for input_layer=embed
    """

    def __init__(self, idim,
                 attention_dim=256,
                 attention_heads=4,
                 linear_units=2048,
                 num_blocks=6,
                 dropout_rate=0.1,
                 positional_dropout_rate=0.1,
                 attention_dropout_rate=0.0,
                 input_layer="conv2d",
                 pos_enc_class=PositionalEncoding,
                 normalize_before=True,
                 concat_after=False,
                 positionwise_layer_type="linear",
                 positionwise_conv_kernel_size=1,
                 padding_idx=-1,
                 ali_file=None,
                 ali_sampled=False,
                 aux_pos=None,
                 aux_only=False):  # aux_pos including: FB(FBank), COVOUT(Cov_out), ENOUT(Encoder_out)
        """Construct an Encoder object."""
        super(Encoder, self).__init__()


        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(idim, attention_dim),
                torch.nn.LayerNorm(attention_dim),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(attention_dim, positional_dropout_rate)
            )
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(idim, attention_dim, dropout_rate)
        elif input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(idim, attention_dim, padding_idx=padding_idx),
                pos_enc_class(attention_dim, positional_dropout_rate)
            )
        elif isinstance(input_layer, torch.nn.Module):
            self.embed = torch.nn.Sequential(
                input_layer,
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer is None:
            self.embed = torch.nn.Sequential(
                pos_enc_class(attention_dim, positional_dropout_rate)
            )
        else:
            raise ValueError("unknown input_layer: " + input_layer)
        self.normalize_before = normalize_before
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (attention_dim, linear_units, dropout_rate)
        elif positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (attention_dim, linear_units, positionwise_conv_kernel_size, dropout_rate)
        elif positionwise_layer_type == "conv1d-linear":
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (attention_dim, linear_units, positionwise_conv_kernel_size, dropout_rate)
        else:
            raise NotImplementedError("Support only linear or conv1d.")

        # auxilary model relevant begin
        # only when aux_model_path is not None, the three attribute below is meaningful
        print('aux_pos=', aux_pos, 'aux_only=', aux_only, 'ali_file=', ali_file, 'ali_sampled = ',ali_sampled,flush=True)

        self.aux_model = AuxModel(ali_file, ali_sampled)
        self.aux_model.eval()
        self.aux_pos = aux_pos
        self.aux_only = aux_only
        if aux_only:
            assert self.aux_pos is not None, "aux_only is True, but aux_pos is None"
            self.aux_linear = torch.nn.Linear(self.aux_model.dim, attention_dim)
        else:
            self.aux_linear = torch.nn.Linear(attention_dim+self.aux_model.dim, attention_dim)

        self.encoders = repeat(
            num_blocks,
            lambda: EncoderLayer(
                attention_dim,
                MultiHeadedAttention(attention_heads, attention_dim, attention_dropout_rate),
                positionwise_layer(*positionwise_layer_args),
                dropout_rate,
                normalize_before,
                concat_after
            )
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)

    def forward(self, xs, masks, uttid_list, train):
        """Embed positions in tensor.
           :param torch.Tensor xs: input tensor
           :param torch.Tensor masks: input mask
           :return: position embedded tensor and mask
           :rtype Tuple[torch.Tensor, torch.Tensor]:
        """
        # xs (b, t, f) 

        if isinstance(self.embed, Conv2dSubsampling):
            with torch.no_grad():
                aux_emb = self.aux_model(uttid_list, xs.shape[1], xs.device, True) # (b, t, bn)
            xs, masks = self.embed(xs, masks)
        else:
            with torch.no_grad():
                aux_emb = self.aux_model(uttid_list, xs.shape[1], xs.device, False) # (b, t, bn)
            xs = self.embed(xs)

        if self.aux_pos == "COVOUT":
            if self.aux_only:
                xs = aux_emb
            else:
                xs = torch.cat((xs, aux_emb), dim=2)
            xs = self.aux_linear(xs) # (b, t, n_att+bn) -> (b,t,n_att)

        xs, masks = self.encoders(xs, masks)

        if self.aux_pos == "ENOUT":
            if self.aux_only:
                xs = aux_emb
            else:
                xs = torch.cat((xs, aux_emb), dim=2)
            xs = self.aux_linear(xs) # (b, t, n_att+bn) -> (b,t,n_att)

        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks


# test class
if __name__ == '__main__':
    import torch
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import numpy as np
    np.random.seed(0)

    #aux_model_path = '/mnt/lustre/sjtu/users/mkh96/wordspace/asr/codeswitch/exp/phone_classifier/transformer_layer12_lsm0.0_ep100/results/snapshot.ep.100'
    encoder = Encoder(
        idim=80,
        attention_dim=4,
        attention_heads=4,
        linear_units=3,
        num_blocks=2,
        input_layer="conv2d",
        dropout_rate=0,
        positional_dropout_rate=0,
        attention_dropout_rate=0,
        aux_pos="COVOUT")  # aux_pos including:COVOUT(Cov_out), ENOUT(Encoder_out)

    itensor = torch.randn(2, 400, 80)
    uttid_list = ['cn500h1_T0055G0002S0001', 'cn500h1_T0055G0002S0002' ]
    #print('input tensor', itensor)
    masks = torch.ones(2, 1, 400)
    masks[1][0][200:] = 0 # (b, c, l)
    #print('masks', masks)
    otensor, masks = encoder(itensor, masks, uttid_list)
    print('output tensor', otensor, masks, otensor.size())
