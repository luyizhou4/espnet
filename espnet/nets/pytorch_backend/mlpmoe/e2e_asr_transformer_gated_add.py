# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer speech recognition model (pytorch)."""

from argparse import Namespace
from distutils.util import strtobool

import logging
import math

import torch

from espnet.nets.asr_interface import ASRInterface
from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.e2e_asr import CTC_LOSS_THRESHOLD
from espnet.nets.pytorch_backend.e2e_asr import Reporter
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.initializer import initialize
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import LabelSmoothingLoss
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet.nets.pytorch_backend.transformer.plot import PlotAttentionReport

# encoder output proj-layer layer norm, self-defined LN changes eps from 1e-5 to 1e-12
# we just keep this usage as initial espnet setup
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm

class E2E(ASRInterface, torch.nn.Module):
    """E2E module.

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options

    """

    @staticmethod
    def add_arguments(parser):
        """Add arguments."""
        group = parser.add_argument_group("transformer model setting")

        group.add_argument("--transformer-init", type=str, default="pytorch",
                           choices=["pytorch", "xavier_uniform", "xavier_normal",
                                    "kaiming_uniform", "kaiming_normal"],
                           help='how to initialize transformer parameters')
        group.add_argument("--transformer-input-layer", type=str, default="conv2d",
                           choices=["conv2d", "linear", "embed"],
                           help='transformer input layer type')
        group.add_argument('--transformer-attn-dropout-rate', default=None, type=float,
                           help='dropout in transformer attention. use --dropout-rate if None is set')
        group.add_argument('--transformer-lr', default=10.0, type=float,
                           help='Initial value of learning rate')
        group.add_argument('--transformer-warmup-steps', default=25000, type=int,
                           help='optimizer warmup steps')
        group.add_argument('--transformer-length-normalized-loss', default=True, type=strtobool,
                           help='normalize loss by length')

        group.add_argument('--dropout-rate', default=0.0, type=float,
                           help='Dropout rate for the encoder')
        # Encoder
        group.add_argument('--elayers', default=4, type=int,
                           help='Number of encoder layers (for shared recognition part in multi-speaker asr mode)')
        group.add_argument('--eunits', '-u', default=300, type=int,
                           help='Number of encoder hidden units')
        # Attention
        group.add_argument('--adim', default=320, type=int,
                           help='Number of attention transformation dimensions')
        group.add_argument('--aheads', default=4, type=int,
                           help='Number of heads for multi head attention')
        # Decoder
        group.add_argument('--dlayers', default=1, type=int,
                           help='Number of decoder layers')
        group.add_argument('--dunits', default=320, type=int,
                           help='Number of decoder hidden units')
        # ctc init path
        group.add_argument('--pretrained-cn-ctc-model', default='', type=str,
                         help='pretrained cn ctc model')
        group.add_argument('--pretrained-en-ctc-model', default='', type=str,
                         help='pretrained en ctc model')
        group.add_argument('--pretrained-mlme-model', default='', type=str,
                         help='pretrained multi-lingual multi-encoder model')
        # gated-add lambda_ vectorization
        group.add_argument('--vectorize-lambda', default=False, type=strtobool,
                         help='vectorize lambda for encoder fusion')
        # this is used to keep mandarin performance
        group.add_argument('--pretrained-cn-jca-model', default='', type=str,
                         help='pretrained cn jca model')
        return parser

    @property
    def attention_plot_class(self):
        """Return PlotAttentionReport."""
        return PlotAttentionReport

    def __init__(self, idim, odim, args, ignore_id=-1):
        """Construct an E2E object.

        :param int idim: dimension of inputs
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        torch.nn.Module.__init__(self)
        if args.transformer_attn_dropout_rate is None:
            args.transformer_attn_dropout_rate = args.dropout_rate
        self.cn_encoder = Encoder(
            idim=idim,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.eunits,
            num_blocks=args.elayers,
            input_layer=args.transformer_input_layer,
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            attention_dropout_rate=args.transformer_attn_dropout_rate
        )
        self.en_encoder = Encoder(
            idim=idim,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.eunits,
            num_blocks=args.elayers,
            input_layer=args.transformer_input_layer,
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            attention_dropout_rate=args.transformer_attn_dropout_rate
        )
        # gated add module 
        self.vectorize_lambda = args.vectorize_lambda
        lambda_dim = args.adim if self.vectorize_lambda else 1
        self.aggregation_module = torch.nn.Sequential(
                torch.nn.Linear(2*args.adim, lambda_dim),
                torch.nn.Sigmoid())

        self.decoder = Decoder(
            odim=odim,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.dunits,
            num_blocks=args.dlayers,
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            self_attention_dropout_rate=args.transformer_attn_dropout_rate,
            src_attention_dropout_rate=args.transformer_attn_dropout_rate
        )
        self.sos = odim - 1
        self.eos = odim - 1
        self.odim = odim
        self.ignore_id = ignore_id
        self.subsample = [1]
        self.reporter = Reporter()

        # self.lsm_weight = a
        self.criterion = LabelSmoothingLoss(self.odim, self.ignore_id, args.lsm_weight,
                                            args.transformer_length_normalized_loss)
        # self.verbose = args.verbose
        self.adim = args.adim
        self.mtlalpha = args.mtlalpha
        if args.mtlalpha > 0.0:
            self.ctc = CTC(odim, args.adim, args.dropout_rate, ctc_type=args.ctc_type, reduce=True)
        else:
            self.ctc = None

        if args.report_cer or args.report_wer:
            from espnet.nets.e2e_asr_common import ErrorCalculator
            self.error_calculator = ErrorCalculator(args.char_list,
                                                    args.sym_space, args.sym_blank,
                                                    args.report_cer, args.report_wer)
        else:
            self.error_calculator = None
        self.rnnlm = None

        # yzl23 config
        self.remove_blank_in_ctc_mode = True
        self.reset_parameters(args) # reset params at the last

        logging.warning("Model total size: {}M, requires_grad size: {}M"
                .format(self.count_parameters(), self.count_parameters(requires_grad=True)))

    def count_parameters(self, requires_grad=False):
        if requires_grad:
            return sum(p.numel() for p in self.parameters() if p.requires_grad) / 1024 / 1024
        else:
            return sum(p.numel() for p in self.parameters()) / 1024 / 1024

    def reset_parameters(self, args):
        """Initialize parameters."""
        
        # load state_dict, and keeps only encoder part
        # note that self.ctc.ctc_lo is also removed
        # prefix is added to meet the needs of moe structure
        def load_state_dict_encoder(path, prefix=''):
            if 'snapshot' in path:
                model_state_dict = torch.load(path, map_location=lambda storage, loc: storage)['model']
            else:
                model_state_dict = torch.load(path, map_location=lambda storage, loc: storage)
            for k in list(model_state_dict.keys()):
                if not 'encoder' in k:
                    # remove this key
                    del model_state_dict[k]
                else:
                    new_k = k.replace('encoder.', prefix + 'encoder.')
                    model_state_dict[new_k] = model_state_dict.pop(k)
            return model_state_dict
        
        def load_state_dict_all(path, prefix=''):
            if 'snapshot' in path:
                model_state_dict = torch.load(path, map_location=lambda storage, loc: storage)['model']
            else:
                model_state_dict = torch.load(path, map_location=lambda storage, loc: storage)
            for k in list(model_state_dict.keys()):
                if not 'encoder' in k:
                    continue
                else:
                    new_k = k.replace('encoder.', prefix + 'encoder.')
                    model_state_dict[new_k] = model_state_dict.pop(k)
            return model_state_dict
        
        # initialize parameters
        if args.pretrained_mlme_model:
            logging.warning("loading pretrained mlme model for parallel encoder")
            # still need to initialize the 'other' params
            initialize(self, args.transformer_init)
            path = args.pretrained_mlme_model
            if 'snapshot' in path:
                model_state_dict = torch.load(path, map_location=lambda storage, loc: storage)['model']
            else:
                model_state_dict = torch.load(path, map_location=lambda storage, loc: storage)
            self.load_state_dict(model_state_dict, strict=False)
            del model_state_dict
        elif args.pretrained_cn_ctc_model and args.pretrained_en_ctc_model:
            logging.warning("loading pretrained ctc model for parallel encoder")
            # still need to initialize the 'other' params
            initialize(self, args.transformer_init)
            cn_state_dict = load_state_dict_encoder(args.pretrained_cn_ctc_model, 
                                                        prefix='cn_')
            self.load_state_dict(cn_state_dict, strict=False)
            del cn_state_dict
            en_state_dict = load_state_dict_encoder(args.pretrained_en_ctc_model, 
                                                        prefix='en_')
            self.load_state_dict(en_state_dict, strict=False)
            del en_state_dict
        elif args.pretrained_cn_jca_model and args.pretrained_en_ctc_model:
            logging.warning("loading pretrained cn-jca & en-ctc model for parallel encoder")
            initialize(self, args.transformer_init)
            en_state_dict = load_state_dict_encoder(args.pretrained_en_ctc_model, 
                                                        prefix='en_')
            self.load_state_dict(en_state_dict, strict=False)
            del en_state_dict
            cn_state_dict = load_state_dict_all(args.pretrained_cn_jca_model, 
                                                        prefix='cn_')
            self.load_state_dict(cn_state_dict, strict=False)
            del cn_state_dict
        else:
            initialize(self, args.transformer_init)


    def forward(self, xs_pad, ilens, ys_pad):
        """E2E forward.

        :param torch.Tensor xs_pad: batch of padded source sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of source sequences (B)
        :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
        :return: ctc loass value
        :rtype: torch.Tensor
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy in attention decoder
        :rtype: float
        """
        # 1. forward encoder
        xs_pad = xs_pad[:, :max(ilens)]  # for data parallel
        src_mask = (~make_pad_mask(ilens.tolist())).to(xs_pad.device).unsqueeze(-2)
        # mlp moe forward
        cn_hs_pad, hs_mask = self.cn_encoder(xs_pad, src_mask)
        en_hs_pad, hs_mask = self.en_encoder(xs_pad, src_mask)
        # gated add module 
        """ lambda = sigmoid(W_cn * cn_xs + w_en * en_xs + b)  #(B, T, 1)
            xs = lambda * cn_xs + (1-lambda) * en_xs 
        """
        hs_pad = torch.cat((cn_hs_pad, en_hs_pad), dim=-1)
        lambda_ = self.aggregation_module(hs_pad) # (B,T,1)/(B,T,D), range from (0, 1)
        hs_pad = lambda_ * cn_hs_pad + (1 - lambda_) * en_hs_pad
        self.hs_pad = hs_pad

        # TODO(karita) show predicted text
        # TODO(karita) calculate these stats
        cer_ctc = None
        if self.mtlalpha == 0.0:
            loss_ctc = None
        else:
            batch_size = xs_pad.size(0)
            hs_len = hs_mask.view(batch_size, -1).sum(1)
            loss_ctc = self.ctc(hs_pad.view(batch_size, -1, self.adim), hs_len, ys_pad)
            if self.error_calculator is not None:
                ys_hat = self.ctc.argmax(hs_pad.view(batch_size, -1, self.adim)).data
                cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)

        if self.mtlalpha == 1:
            self.loss_att, acc = None, None
        else:
            # 2. forward decoder
            ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
            ys_mask = target_mask(ys_in_pad, self.ignore_id)
            pred_pad, pred_mask = self.decoder(ys_in_pad, ys_mask, hs_pad, hs_mask)
            self.pred_pad = pred_pad

            # 3. compute attention loss
            loss_att = self.criterion(pred_pad, ys_out_pad)
            acc = th_accuracy(pred_pad.view(-1, self.odim), ys_out_pad,
                                   ignore_label=self.ignore_id)
        self.acc = acc

        # 5. compute cer/wer
        if self.training or self.error_calculator is None:
            cer, wer = None, None
        else:
            ys_hat = pred_pad.argmax(dim=-1)
            cer, wer = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        # copyied from e2e_asr
        alpha = self.mtlalpha
        if alpha == 0:
            self.loss = loss_att
            loss_att_data = float(loss_att)
            loss_ctc_data = None
        elif alpha == 1:
            self.loss = loss_ctc
            loss_att_data = None
            loss_ctc_data = float(loss_ctc)
        else:
            self.loss = alpha * loss_ctc + (1 - alpha) * loss_att
            loss_att_data = float(loss_att)
            loss_ctc_data = float(loss_ctc)

        loss_data = float(self.loss)
        if loss_data < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
            self.reporter.report(loss_ctc_data, loss_att_data, self.acc, cer_ctc, cer, wer, loss_data)
        else:
            logging.warning('loss (=%f) is not correct', loss_data)
        return self.loss

    def encode(self, x):
        """Encode acoustic features.

        :param ndarray x: source acoustic feature (T, D)
        :return: encoder outputs
        :rtype: torch.Tensor
        """
        self.eval()
        x = torch.as_tensor(x).unsqueeze(0) # (B, T, D) with #B=1
        cn_enc_output, _ = self.cn_encoder(x, None)
        en_enc_output, _ = self.en_encoder(x, None)
        enc_output = torch.cat((cn_enc_output, en_enc_output), dim=-1)
        lambda_ = self.aggregation_module(enc_output) # (B,T,1), range from (0, 1)
        enc_output = lambda_ * cn_enc_output + (1 - lambda_) * en_enc_output
        
        # inverse lambda decode 
        # first round lambda_ params to 0/1
        # lambda_ = lambda_.round()
        # enc_output = (1 - lambda_) * cn_enc_output + lambda_ * en_enc_output
        
        return enc_output.squeeze(0) # returns tensor(T, D)

    def recognize(self, x, recog_args, char_list=None, rnnlm=None, use_jit=False):
        if recog_args.ctc_greedy_decoding:
            return self.recognize_ctc_greedy(x, recog_args)
        else:
            return self.recognize_jca(x, recog_args, char_list, rnnlm, use_jit)

    def store_penultimate_state(self, xs_pad, ilens, ys_pad):
        xs_pad = xs_pad[:, :max(ilens)]  # for data parallel
        src_mask = (~make_pad_mask(ilens.tolist())).to(xs_pad.device).unsqueeze(-2)
        # multi-encoder forward
        cn_hs_pad, hs_mask = self.cn_encoder(xs_pad, src_mask)
        en_hs_pad, hs_mask = self.en_encoder(xs_pad, src_mask)
        hs_pad = torch.cat((cn_hs_pad, en_hs_pad), dim=-1)

        lambda_ = self.aggregation_module(hs_pad) # (B,T,1), range from (0, 1)
        hs_pad = lambda_ * cn_hs_pad + (1 - lambda_) * en_hs_pad
        penultimate_state = lambda_
        # self.hs_pad = hs_pad

        # forward decoder
        # ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        # ys_mask = target_mask(ys_in_pad, self.ignore_id)
        # pred_pad, pred_mask, penultimate_state = self.decoder(ys_in_pad, ys_mask, hs_pad, hs_mask, return_penultimate_state=True)

        # plot penultimate_state, (B,T,att_dim)
        return penultimate_state.squeeze(0).detach().cpu().numpy()

    def recognize_ctc_greedy(self, x, recog_args):
        """Recognize input speech with ctc greedy decoding.

        :param ndnarray x: input acoustic feature (B, T, D) or (T, D)
        :param Namespace recog_args: argment Namespace contraining options
        :return: N-best decoding results (fake results for compatibility)
        :rtype: list
        """
        enc_output = self.encode(x).unsqueeze(0) # (1, T, D)
        lpz = self.ctc.log_softmax(enc_output)
        lpz = lpz.squeeze(0) # shape of (T, D)
        idx = lpz.argmax(-1).cpu().numpy().tolist()
        hyp = {}
        if recog_args.ctc_raw_results:
            hyp['yseq'] = [self.sos] + idx # not apply ctc mapping, to get ctc alignment
        else:
            # <sos> is added here to be compatible with S2S decoding, 
            # file: espnet/asr/asr_utils/parse_hypothesis
            hyp['yseq'] = [self.sos] + self.ctc_mapping(idx)
        logging.info(hyp['yseq'])
        hyp['score'] = -1
        return [hyp]

    def ctc_mapping(self, x, blank=0):
        prev = blank
        y = []
        for i in x:
            if i != blank and i != prev:
                y.append(i)
            prev = i
        return y

    def recognize_jca(self, x, recog_args, char_list=None, rnnlm=None, use_jit=False):
        """Recognize input speech.

        :param ndnarray x: input acoustic feature (B, T, D) or (T, D)
        :param Namespace recog_args: argment Namespace contraining options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        enc_output = self.encode(x).unsqueeze(0) # (1, T, D)
        if recog_args.ctc_weight > 0.0:
            lpz = self.ctc.log_softmax(enc_output)
            lpz = lpz.squeeze(0) # shape of (T, D)
        else:
            lpz = None

        h = enc_output.squeeze(0) # (B, T, D), #B=1

        logging.info('input lengths: ' + str(h.size(0)))
        # search parms
        beam = recog_args.beam_size
        penalty = recog_args.penalty
        ctc_weight = recog_args.ctc_weight

        # preprare sos
        y = self.sos
        vy = h.new_zeros(1).long()

        if recog_args.maxlenratio == 0:
            maxlen = h.shape[0]
        else:
            # maxlen >= 1
            maxlen = max(1, int(recog_args.maxlenratio * h.size(0)))
        minlen = int(recog_args.minlenratio * h.size(0))
        logging.info('max output length: ' + str(maxlen))
        logging.info('min output length: ' + str(minlen))

        # initialize hypothesis
        if rnnlm:
            hyp = {'score': 0.0, 'yseq': [y], 'rnnlm_prev': None}
        else:
            hyp = {'score': 0.0, 'yseq': [y]}
        if lpz is not None:
            import numpy

            from espnet.nets.ctc_prefix_score import CTCPrefixScore

            ctc_prefix_score = CTCPrefixScore(lpz.detach().numpy(), 0, self.eos, numpy)
            hyp['ctc_state_prev'] = ctc_prefix_score.initial_state()
            hyp['ctc_score_prev'] = 0.0
            if ctc_weight != 1.0:
                # pre-pruning based on attention scores
                from espnet.nets.pytorch_backend.rnn.decoders import CTC_SCORING_RATIO
                ctc_beam = min(lpz.shape[-1], int(beam * CTC_SCORING_RATIO))
            else:
                if self.remove_blank_in_ctc_mode:
                    ctc_beam = lpz.shape[-1] - 1 # except blank
                else:
                    ctc_beam = lpz.shape[-1]
        hyps = [hyp]
        ended_hyps = []

        import six
        traced_decoder = None
        for i in six.moves.range(maxlen):
            logging.debug('position ' + str(i))

            hyps_best_kept = []
            for hyp in hyps:
                vy.unsqueeze(1)
                vy[0] = hyp['yseq'][i]

                # get nbest local scores and their ids
                ys_mask = subsequent_mask(i + 1).unsqueeze(0) # mask scores of future state
                ys = torch.tensor(hyp['yseq']).unsqueeze(0)
                # FIXME: jit does not match non-jit result
                if use_jit:
                    if traced_decoder is None:
                        traced_decoder = torch.jit.trace(self.decoder.forward_one_step,
                                                         (ys, ys_mask, enc_output))
                    local_att_scores = traced_decoder(ys, ys_mask, enc_output)[0]
                else:
                    local_att_scores = self.decoder.forward_one_step(ys, ys_mask, enc_output)[0]
                if rnnlm:
                    rnnlm_state, local_lm_scores = rnnlm.predict(hyp['rnnlm_prev'], vy)
                    local_scores = local_att_scores + recog_args.lm_weight * local_lm_scores
                else:
                    local_scores = local_att_scores

                if lpz is not None:
                    if self.remove_blank_in_ctc_mode:
                        # here we need to filter out <blank> in local_best_ids
                        # it happens in pure ctc-mode, when ctc_beam equals to #vocab
                        local_best_scores, local_best_ids = torch.topk(
                            local_att_scores[:, 1:], ctc_beam, dim=1)
                        local_best_ids += 1 # hack 
                    else:
                        local_best_scores, local_best_ids = torch.topk(
                            local_att_scores, ctc_beam, dim=1)
                    ctc_scores, ctc_states = ctc_prefix_score(
                        hyp['yseq'], local_best_ids[0], hyp['ctc_state_prev'])
                    local_scores = \
                        (1.0 - ctc_weight) * local_att_scores[:, local_best_ids[0]] \
                        + ctc_weight * torch.from_numpy(ctc_scores - hyp['ctc_score_prev'])
                    if rnnlm:
                        local_scores += recog_args.lm_weight * local_lm_scores[:, local_best_ids[0]]
                    local_best_scores, joint_best_ids = torch.topk(local_scores, beam, dim=1)
                    local_best_ids = local_best_ids[:, joint_best_ids[0]]
                else:
                    local_best_scores, local_best_ids = torch.topk(local_scores, beam, dim=1)

                for j in six.moves.range(beam):
                    new_hyp = {}
                    new_hyp['score'] = hyp['score'] + float(local_best_scores[0, j])
                    new_hyp['yseq'] = [0] * (1 + len(hyp['yseq']))
                    new_hyp['yseq'][:len(hyp['yseq'])] = hyp['yseq']
                    new_hyp['yseq'][len(hyp['yseq'])] = int(local_best_ids[0, j])
                    if rnnlm:
                        new_hyp['rnnlm_prev'] = rnnlm_state
                    if lpz is not None:
                        new_hyp['ctc_state_prev'] = ctc_states[joint_best_ids[0, j]]
                        new_hyp['ctc_score_prev'] = ctc_scores[joint_best_ids[0, j]]
                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)

                hyps_best_kept = sorted(
                    hyps_best_kept, key=lambda x: x['score'], reverse=True)[:beam]

            # sort and get nbest
            hyps = hyps_best_kept
            logging.debug('number of pruned hypothes: ' + str(len(hyps)))
            if char_list is not None:
                logging.debug(
                    'best hypo: ' + ''.join([char_list[int(x)] for x in hyps[0]['yseq'][1:]]))

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                logging.info('adding <eos> in the last postion in the loop')
                for hyp in hyps:
                    hyp['yseq'].append(self.eos)

            # add ended hypothes to a final list, and removed them from current hypothes
            # (this will be a probmlem, number of hyps < beam)
            remained_hyps = []
            for hyp in hyps:
                if hyp['yseq'][-1] == self.eos:
                    # only store the sequence that has more than minlen outputs
                    # also add penalty
                    if len(hyp['yseq']) > minlen:
                        hyp['score'] += (i + 1) * penalty
                        if rnnlm:  # Word LM needs to add final <eos> score
                            hyp['score'] += recog_args.lm_weight * rnnlm.final(
                                hyp['rnnlm_prev'])
                        ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            # end detection
            # from espnet.nets.e2e_asr_common import end_detect
            # if end_detect(ended_hyps, i) and recog_args.maxlenratio == 0.0:
            from espnet.nets.e2e_asr_common import end_detect_yzl23
            if end_detect_yzl23(ended_hyps, remained_hyps, penalty) and recog_args.maxlenratio == 0.0:
                logging.info('end detected at %d', i)
                break

            hyps = remained_hyps
            if len(hyps) > 0:
                logging.debug('remeined hypothes: ' + str(len(hyps)))
            else:
                logging.info('no hypothesis. Finish decoding.')
                break

            if char_list is not None:
                for hyp in hyps:
                    logging.debug(
                        'hypo: ' + ''.join([char_list[int(x)] for x in hyp['yseq'][1:]]))

            logging.debug('number of ended hypothes: ' + str(len(ended_hyps)))

        nbest_hyps = sorted(
            ended_hyps, key=lambda x: x['score'], reverse=True)[:min(len(ended_hyps), recog_args.nbest)]

        # check number of hypotheis
        if len(nbest_hyps) == 0:
            logging.warning('there is no N-best results, perform recognition again with smaller minlenratio.')
            # should copy becasuse Namespace will be overwritten globally
            recog_args = Namespace(**vars(recog_args))
            recog_args.minlenratio = max(0.0, recog_args.minlenratio - 0.1)
            return self.recognize(x, recog_args, char_list, rnnlm)

        logging.info('total log probability: ' + str(nbest_hyps[0]['score']))
        logging.info('normalized log probability: ' + str(nbest_hyps[0]['score'] / len(nbest_hyps[0]['yseq'])))
        return nbest_hyps

    def calculate_all_attentions(self, xs_pad, ilens, ys_pad):
        """E2E attention calculation.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :return: attention weights with the following shape,
            1) multi-head case => attention weights (B, H, Lmax, Tmax),
            2) other case => attention weights (B, Lmax, Tmax).
        :rtype: float ndarray
        """
        with torch.no_grad():
            self.forward(xs_pad, ilens, ys_pad)
        ret = dict()
        for name, m in self.named_modules():
            if isinstance(m, MultiHeadedAttention):
                ret[name] = m.attn.cpu().numpy()
        return ret
