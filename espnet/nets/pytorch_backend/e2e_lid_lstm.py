#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""RNN sequence-to-sequence speech recognition model (pytorch)."""

from __future__ import division

import argparse
import logging
import math
import os

import chainer
import numpy as np
import six
import torch

from chainer import reporter
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


from espnet.nets.asr_interface import ASRInterface
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import LabelSmoothingLoss

CTC_LOSS_THRESHOLD = 10000


class Reporter(chainer.Chain):
    """A chainer reporter wrapper."""

    def report(self, acc, loss):
        """Report at every step."""
        reporter.report({'acc': acc}, self)
        reporter.report({'loss': loss}, self)

class CNN(torch.nn.Module):
    def __init__(self, idim, eunits):
        super(CNN, self).__init__()
        logging.warning('we force to make cnn out-channels 256')
        oc = 256
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, oc, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(oc, oc, 3, 2),
            torch.nn.ReLU()
        )
        self.proj = torch.nn.Linear(oc * (((idim - 1) // 2 - 1) // 2), eunits)

    def forward(self, xs_pad, ilens):
        xs_pad = xs_pad.view(xs_pad.size(0), xs_pad.size(1), 1, xs_pad.size(2)).transpose(1, 2)
        xs_pad = self.conv(xs_pad)
        if torch.is_tensor(ilens):
            ilens = ilens.cpu().numpy()
            ilens = np.array(ilens, dtype=np.float32)
        else:
            ilens = np.array(ilens, dtype=np.float32)
        ilens = np.array(np.floor((ilens-1) / 2), dtype=np.int64)
        ilens = np.array(
            np.floor( (np.array(ilens, dtype=np.float32)-1) / 2), dtype=np.int64).tolist()
        # x: utt_list of frame (remove zeropaded frames) x (input channel num x dim)
        xs_pad = xs_pad.transpose(1, 2)
        xs_pad = xs_pad.contiguous().view(
            xs_pad.size(0), xs_pad.size(1), xs_pad.size(2) * xs_pad.size(3))
        # proj layer
        xs_pad = self.proj(xs_pad)
        return xs_pad, ilens


class LSTM(torch.nn.Module):
    def __init__(self, eunits, elayers, dropout):
        super(LSTM, self).__init__()
        self.lstm = torch.nn.LSTM(eunits, eunits, num_layers=elayers, bidirectional=True, batch_first=True, dropout=dropout)
        self.proj = torch.nn.Linear(2 * eunits, eunits, bias=False)
        self.dropout_layer = torch.nn.Dropout(p=dropout)

    def forward(self, xs_pad, ilens):
        # input: [B, T, H]
        xs_pad = pack_padded_sequence(xs_pad, ilens, batch_first=True)
        self.lstm.flatten_parameters()
        xs_pad, _ = self.lstm(xs_pad)
        xs_pad, _ = pad_packed_sequence(xs_pad, batch_first=True)
        xs_pad = self.proj(self.dropout_layer(xs_pad))
        return xs_pad, ilens


class E2E(ASRInterface, torch.nn.Module):
    """E2E module.

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options

    """

    @staticmethod
    def add_arguments(parser):
        """Add arguments for the encoder."""
        group = parser.add_argument_group("E2E encoder setting")
        # encoder
        group.add_argument('--etype', default='cnnblstm', type=str,
                           help='Type of encoder network architecture')
        group.add_argument('--elayers', default=3, type=int,
                           help='Number of encoder layers (for shared recognition part in multi-speaker asr mode)')
        group.add_argument('--eunits', '-u', default=512, type=int,
                           help='Number of encoder hidden units')
        group.add_argument('--dropout-rate', default=0.2, type=float,
                           help='Dropout rate for the encoder')
        return parser

    def __init__(self, idim, odim, args):
        """Construct an E2E object.

        :param int idim: dimension of inputs
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        super(E2E, self).__init__()
        torch.nn.Module.__init__(self)
        self.reporter = Reporter()
        self.subsample = [1]
        self.ignore_id = -1
        self.idim = idim
        self.odim = odim

        # conv front-end, compatible with transformer structure
        self.conv = CNN(idim, args.eunits)
        self.lstm = LSTM(args.eunits, args.elayers, args.dropout_rate)
        self.lid_lo = torch.nn.Linear(args.eunits, odim)

        self.criterion = LabelSmoothingLoss(odim, self.ignore_id, smoothing=args.lsm_weight,
                                            normalize_length=False)
        # weight initialization
        self.reset_parameters()
        self.loss = None
        self.acc = None
        logging.warning(self)

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if len(param.shape) > 1: 
                torch.nn.init.xavier_uniform_(param.data)
            if 'bias' in name:
                param.data.zero_()
            # init lstm forget gate bias
            if ('lstm' in name or 'rnn' in name) and 'bias' in name:
                n = param.size(0) 
                param.data[n // 4:n // 2].fill_(1.)

    def forward(self, xs_pad, ilens, ys_pad):
        """E2E forward.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :return: loss value
        :rtype: torch.Tensor
        """
        # 1. Encoder
        xs_pad, ilens = self.conv(xs_pad, ilens)
        xs_pad, ilens = self.lstm(xs_pad, ilens)
        pred_pad = self.lid_lo(xs_pad)

        # compute lid loss
        self.loss = self.criterion(pred_pad, ys_pad)
        self.acc = th_accuracy(pred_pad.view(-1, self.odim), ys_pad,
                               ignore_label=self.ignore_id)

        loss_data = float(self.loss)
        if loss_data < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
            self.reporter.report(self.acc, loss_data)
        else:
            logging.warning('loss (=%f) is not correct', loss_data)
        return self.loss

    def encode(self, x):
        """Encode acoustic features.

        :param ndarray x: input acoustic feature (T, D)
        :return: encoder outputs
        :rtype: torch.Tensor
        """
        self.eval()
        ilens = [x.shape[0]]

        # subsample frame
        x = x[::self.subsample[0], :]
        p = next(self.parameters())
        h = torch.as_tensor(x, device=p.device, dtype=p.dtype)
        # make a utt list (1) to use the same interface for encoder
        hs = h.contiguous().unsqueeze(0)

        # 1. encoder
        hs, hlens = self.conv(hs, ilens)
        hs, hlens = self.lstm(hs, ilens)
        return hs.squeeze(0)

    def recognize(self, x, recog_args, char_list=None, rnnlm=None):
        """E2E beam search.

        :param ndarray x: input acoustic feature (T, D)
        :param Namespace recog_args: argument Namespace containing options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        self.eval()
        enc_output = self.encode(x).unsqueeze(0) # (1, T, D)
        lpz = self.lid_lo(enc_output)
        lpz = lpz.squeeze(0) # shape of (T, D)
        idx = lpz.argmax(-1).cpu().numpy().tolist()
        hyp = {}
        # [-1] is added here to be compatible with S2S decoding, 
        # file: espnet/asr/asr_utils/parse_hypothesis
        hyp['yseq'] = [-1] + idx # not apply ctc mapping, to get ctc alignment
        logging.info(hyp['yseq'])
        hyp['score'] = -1
        return [hyp]

    # todo: batch processing, currently decoding mix200 with 48 cpu consumes around one hour
    def store_penultimate_state(self, xs_pad, ilens, ys_pad):
        self.eval()
        xs_pad, ilens = self.conv(xs_pad, ilens)
        xs_pad, ilens = self.lstm(xs_pad, ilens)
        # lid output layer
        pred_pad = torch.nn.functional.softmax(self.lid_lo(xs_pad), dim=-1)
        # plot penultimate_state, (B,T,att_dim)
        return pred_pad.squeeze(0).detach().cpu().numpy()

    def calculate_all_attentions(self, xs_pad, ilens, ys_pad):
        return dict()

