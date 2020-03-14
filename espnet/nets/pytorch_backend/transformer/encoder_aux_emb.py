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
from espnet.transform.transformation import Transformation

aux_logger = logging.getLogger('aux_logger')
aux_logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('aux3_onehot.log')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
aux_logger.addHandler(fh)

class AuxModel(torch.nn.Module):
    """Auxiliary Model is built by load trained model and a optional linear Module
    :param string aux_model_path: the path for trained auxiliary model
    :param int aux_n_bn: only when has_linear is True, is meaningful. Used to set the output dim of linear layer 
    :param has_linear: default is False
    :return torch.nn.Module aux_model
    """
    def __init__(self, aux_model_path, onehot=False):
        super(AuxModel, self).__init__()
        aux_trained_model, aux_trained_args = load_trained_model(aux_model_path)
        self.aux_model = aux_trained_model
        #print('aux_model', self.aux_model)
        self.odim = aux_trained_args.adim
        self.onehot = onehot
        if onehot:
            self.odim = 128
    def forward(self, xs, masks):
        emb, masks = self.aux_model.encoder(xs, masks)
        if self.onehot:
            emb = self.aux_model.lid_lo(emb)
            for i in range(len(emb)): # iterate on batch
                    maxi = emb[i].argmax(1)
                    onehot = torch.zeros(emb[i].size(0), self.odim)
                    onehot[torch.arange(emb[i].size(0)), maxi] = 1
                    emb[i] = onehot
        return emb, masks





class Encoder(torch.nn.Module):
    """Transformer encoder module.

    :param int idim: input dim
    :param int attention_dim: dimention of attention
    :param int attention_heads: the number of heads of multi head attention
    :param int linear_units: the number of units of position-wise feed forward
    :par=Noneam int num_blocks: the number of decoder blocks
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
                 aux_model_path=None,
                 onehot=False,
                 aux_pos=None,
                 preprocess_conf=None):  # aux_pos including: FB(FBank), COVOUT(Cov_out), ENOUT(Encoder_out)
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
        print('aux_model_path=', aux_model_path,
              'onehot=', onehot,
              'aux_pos=', aux_pos, flush=True)

        if aux_model_path is not None:
            self.aux_model = AuxModel(aux_model_path, onehot=onehot)
            self.aux_model.eval()
            self.aux_pos = aux_pos
            self.aux_linear = torch.nn.Linear(attention_dim+self.aux_model.odim, attention_dim)

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
        if preprocess_conf is not None:
            self.preprocessing = Transformation(preprocess_conf)            
            logging.warning(
                '[Experimental feature] Some preprocessing will be done '
                'for the mini-batch creation using {} in encoder'
                .format(self.preprocessing))

    def forward(self, xs, masks, uttid_list, train):
        """Embed positions in tensor.
           :param torch.Tensor xs: input tensor
           :param torch.Tensor masks: input mask
           :return: position embedded tensor and mask
           :rtype Tuple[torch.Tensor, torch.Tensor]:
        """

        # xs (b, t, f) 
        if self.aux_model is not None:
            with torch.no_grad():
                aux_emb, aux_masks = self.aux_model(xs, masks) # (b, t, bn)

        # preprocess on encoder input
        # The transformation is procceed in numpy 
        # Dump tensor to numpy and load it back to xs
        # logging.warning('uttid_list={}, train={}'.format(str(uttid_list), str(train)) ) 
        # aux_logger.warning('xs_before {}'.format(str(xs)))
        device = xs.device
        xs_numpy = xs.data.cpu().numpy()
        for i in range(len(xs_numpy)): # iterate on batch
            xs_numpy[i] = self.preprocessing(xs_numpy[i], uttid_list, **{'train': train})
        xs = torch.Tensor(xs_numpy).to(device)
        # aux_logger.warning('xs_after {}'.format(str(xs)))

        # embeding layer
        if isinstance(self.embed, Conv2dSubsampling):
            xs, masks = self.embed(xs, masks)
        else:
            xs = self.embed(xs)
        
        
        if self.aux_pos == "COVOUT":
            xs = torch.cat((xs, aux_emb), dim=2)
            xs = self.aux_linear(xs) # (b, t, n_att+bn) -> (b,t,n_att)

        xs, masks = self.encoders(xs, masks)


        if self.aux_pos == "ENOUT":
            xs = torch.cat((xs, aux_emb), dim=2)
            xs = self.aux_linear(xs) # (b, t, n_att+bn) -> (b,t,n_att)

        if self.normalize_before:
            xs = self.after_norm(xs)
        
        #aux_list=[]
        #for i in range(len(aux_emb[0])):
        #    for j in range(len(aux_emb[0][i])):
        #        if aux_emb[0][i][j] == 1.0:
        #            aux_list.append(j)

        return xs, masks


# test class
if __name__ == '__main__':
    import torch
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import numpy as np
    np.random.seed(0)

    aux_model_path = '/mnt/lustre/sjtu/users/mkh96/wordspace/asr/codeswitch/exp/phone_classifier/transformer_layer12_lsm0.0_ep100/results/snapshot.ep.100'
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
        aux_model_path=aux_model_path,
    aux_has_linear=False,
        aux_n_bn=100,
        aux_pos="ENOUT",
        preprocess_conf='/mnt/lustre/sjtu/home/jqg01/asr/e2e/cs/egs/codeswitching/asr/conf/specaug.yaml')  # aux_pos including:COVOUT(Cov_out), ENOUT(Encoder_out)

    itensor = torch.randn(1, 100, 80)
    #print('input tensor', itensor)
    masks = None #torch.ones(1, 10)
    #print('masks', masks)
    otensor, masks = encoder(itensor, masks, False)
    print('output tensor', otensor, masks, otensor.size())