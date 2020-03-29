#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Label smoothing module."""

import torch
from torch import nn

from espnet.nets.pytorch_backend.nets_utils import pad_list

class LabelSmoothingLoss(nn.Module):
    """Label-smoothing loss.

    :param int size: the number of class
    :param int padding_idx: ignored class id
    :param float smoothing: smoothing rate (0.0 means the conventional CE)
    :param bool normalize_length: normalize loss by sequence length if True
    :param torch.nn.Module criterion: loss function to be smoothed
    """

    def __init__(self, size, padding_idx, smoothing, normalize_length=False, criterion=nn.KLDivLoss(reduce=False)):
        """Construct an LabelSmoothingLoss object."""
        super(LabelSmoothingLoss, self).__init__()
        self.criterion = criterion
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        self.normalize_length = normalize_length

    def forward(self, x, target):
        """Compute loss between x and target.

        :param torch.Tensor x: prediction (batch, seqlen, class)
        :param torch.Tensor target: target signal masked with self.padding_id (batch, seqlen)
        :return: scalar float value
        :rtype torch.Tensor
        """
        assert x.size(2) == self.size
        batch_size = x.size(0)
        x = x.view(-1, self.size)
        target = target.view(-1)
        with torch.no_grad():
            true_dist = x.clone()
            true_dist.fill_(self.smoothing / (self.size - 1))
            ignore = target == self.padding_idx  # (B,)
            total = len(target) - ignore.sum().item()
            target = target.masked_fill(ignore, 0)  # avoid -1 index
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        kl = self.criterion(torch.log_softmax(x, dim=1), true_dist)
        denom = total if self.normalize_length else batch_size
        return kl.masked_fill(ignore.unsqueeze(1), 0).sum() / denom

class MoELanguageIDMultitakLoss(nn.Module):
    """Label-smoothing loss.

    :param int size: the number of class
    :param int padding_idx: ignored class id
    :param float smoothing: smoothing rate (0.0 means the conventional CE)
    :param bool normalize_length: normalize loss by sequence length if True
    :param torch.nn.Module criterion: loss function to be smoothed
    """

    def __init__(self, size, padding_idx, smoothing, normalize_length=False, criterion=nn.KLDivLoss(reduce=False)):
        """Construct an MoELanguageIDMultitakLoss object."""
        super(MoELanguageIDMultitakLoss, self).__init__()
        self.criterion = criterion
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        self.normalize_length = normalize_length

    def forward(self, x, target):
        """Compute loss between x and target.

        :param torch.Tensor x: prediction (batch, seqlen, class)
        :param torch.Tensor target: target signal masked with self.padding_id (batch, seqlen)
        :return: scalar float value
        :rtype torch.Tensor
        """
        assert x.size(2) == self.size
        batch_size = x.size(0)
        x = x.view(-1, self.size)
        target = target.view(-1)
        with torch.no_grad():
            true_dist = x.clone()
            true_dist.fill_(self.smoothing / (self.size - 1))
            ignore = target == self.padding_idx  # (B,)
            total = len(target) - ignore.sum().item()
            target = target.masked_fill(ignore, 0)  # avoid -1 index
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        kl = self.criterion(torch.log(x), true_dist)
        denom = total if self.normalize_length else batch_size
        return kl.masked_fill(ignore.unsqueeze(1), 0).sum() / denom


class LanguageIDMultitakLoss(nn.Module):
    """LanguageIDMultitak loss.
    
    :param int size: the number of class
    :param int padding_idx: ignored class id
    :param bool normalize_length: normalize loss by sequence length if True
    :param torch.nn.Module criterion: loss function to be smoothed
    """

    def __init__(self, padding_idx, normalize_length=False):
        """Construct an LanguageIDMultitakLoss object."""
        super(LanguageIDMultitakLoss, self).__init__()
        self.padding_idx = padding_idx
        reduction = 'mean' if normalize_length else 'sum'
        self.criterion = nn.CrossEntropyLoss(ignore_index=padding_idx, reduction=reduction)

    def forward(self, x, ys_pad):
        """Compute loss between x and target.

        :param torch.Tensor x: prediction (batch, seqlen, class)
        :param torch.Tensor target: target signal masked with self.padding_id (batch, seqlen)
        :return: scalar float value
        :rtype torch.Tensor
        """
        cls_num = x.size(2)
        x = x.view(-1, cls_num)
        target = self.get_actual_target(ys_pad)
        return self.criterion(x, target.view(-1)), target

    def get_actual_target(self, ys_pad, unk_index=1000):
        '''change ys_pad into lid training target

        '''
        ys_list = []
        for y in ys_pad:
            y = y[y != self.padding_idx]
            ylen = y.size(0)
            if ylen > 1: # avoid empty utterance
                y[-1] = y[-2] # hack to set language_id of <eos> to it's previous token
            # now y is a 1-D Tensor
            indicator = y.new_full(y.shape, fill_value=unk_index)
            y = (y < indicator).to(ys_pad)
            ys_list.append(y)

        return pad_list(ys_list, self.padding_idx).to(ys_pad)



