import chainer
from chainer.iterators import MultiprocessIterator
from chainer.iterators import SerialIterator
from chainer.iterators import ShuffleOrderSampler
from chainer.training.extension import Extension

from espnet.utils.training.batchfy import make_batchset
from espnet.utils.dataset import TransformDataset
import logging

import numpy as np

class PerturbSamplingEnabler(Extension):
    """An extension enabling shuffling on an Iterator"""

    def __init__(self, train_iter, train_json, load_tr, converter, args):
        """Inits the PerturbSamplingEnabler

        """
        self.train_iter = train_iter
        self.train_json = train_json
        self.load_tr = load_tr
        self.converter = converter
        self.args = args

    def __call__(self, trainer):
        """Calls the enabler on the given iterator

        :param trainer: The iterator
        """
        args = self.args
        use_sortagrad = args.sortagrad == -1 or args.sortagrad > 0
        # make new batch set for perturb_sampling mode
        # the following are imported from espnet.asr.pytorch_backend.asr:train
        train = make_batchset(self.train_json, args.batch_size,
                          args.maxlen_in, args.maxlen_out, args.minibatches,
                          min_batch_size=args.ngpu if args.ngpu > 1 else 1,
                          shortest_first=use_sortagrad,
                          count=args.batch_count,
                          batch_bins=args.batch_bins,
                          batch_frames_in=args.batch_frames_in,
                          batch_frames_out=args.batch_frames_out,
                          batch_frames_inout=args.batch_frames_inout,
                          iaxis=0, oaxis=0,
                          perturb_sampling=args.perturb_sampling,
                          rank=args.rank,
                          world_size=args.world_size)
        dataset = TransformDataset(train, lambda data: self.converter([self.load_tr(data)]))
        self.train_iter['main'].perturb_sampling_shuffle(dataset)
        logging.warning("Doing Perturb-Sampling shuffling")


class ShufflingEnabler(Extension):
    """An extension enabling shuffling on an Iterator"""

    def __init__(self, iterators):
        """Inits the ShufflingEnabler

        :param list[Iterator] iterators: The iterators to enable shuffling on
        """
        self.set = False
        self.iterators = iterators

    def __call__(self, trainer):
        """Calls the enabler on the given iterator

        :param trainer: The iterator
        """
        if not self.set:
            for iterator in self.iterators:
                iterator.start_shuffle()
            self.set = True


class ToggleableShufflingSerialIterator(SerialIterator):
    """A SerialIterator that can have its shuffling property activated during training"""

    def __init__(self, dataset, batch_size, repeat=True, shuffle=True):
        """Init the Iterator

        :param torch.nn.Tensor dataset: The dataset to take batches from
        :param int batch_size: The batch size
        :param bool repeat: Whether to repeat data (allow multiple epochs)
        :param bool shuffle: Whether to shuffle the batches
        """
        super(ToggleableShufflingSerialIterator, self).__init__(dataset, batch_size, repeat, shuffle)

    def start_shuffle(self):
        """Starts shuffling (or reshuffles) the batches"""
        self._shuffle = True
        if int(chainer._version.__version__[0]) <= 4:
            self._order = np.random.permutation(len(self.dataset))
        else:
            self.order_sampler = ShuffleOrderSampler()
            self._order = self.order_sampler(np.arange(len(self.dataset)), 0)


class ToggleableShufflingMultiprocessIterator(MultiprocessIterator):
    """A MultiprocessIterator that can have its shuffling property activated during training"""

    def __init__(self, dataset, batch_size, repeat=True, shuffle=True, n_processes=None, n_prefetch=1, shared_mem=None,
                 maxtasksperchild=20):
        """Init the iterator

        :param torch.nn.Tensor dataset: The dataset to take batches from
        :param int batch_size: The batch size
        :param bool repeat: Whether to repeat batches or not (enables multiple epochs)
        :param bool shuffle: Whether to shuffle the order of the batches
        :param int n_processes: How many processes to use
        :param int n_prefetch: The number of prefetch to use
        :param int shared_mem: How many memory to share between processes
        :param int maxtasksperchild: Maximum number of tasks per child
        """
        super(ToggleableShufflingMultiprocessIterator, self).__init__(dataset=dataset, batch_size=batch_size,
                                                                      repeat=repeat, shuffle=shuffle,
                                                                      n_processes=n_processes,
                                                                      n_prefetch=n_prefetch, shared_mem=shared_mem,
                                                                      maxtasksperchild=maxtasksperchild)

    def start_shuffle(self):
        """Starts shuffling (or reshuffles) the batches"""
        self.shuffle = True
        if int(chainer._version.__version__[0]) <= 4:
            self._order = np.random.permutation(len(self.dataset))
        else:
            self.order_sampler = ShuffleOrderSampler()
            self._order = self.order_sampler(np.arange(len(self.dataset)), 0)
        self._set_prefetch_state()
