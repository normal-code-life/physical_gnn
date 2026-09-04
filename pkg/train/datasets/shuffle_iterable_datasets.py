"""Buffered shuffling adapter for streaming datasets."""

import random

from torch.utils.data import IterableDataset

from pkg.train.datasets.base_datasets_train import BaseIterableDataset


class ShuffledIterableDataset(BaseIterableDataset):
    """Approximate a global shuffle while retaining bounded memory usage."""

    def __init__(self, dataset, buffer_size):
        """Wrap ``dataset`` with a shuffle buffer of ``buffer_size`` samples."""
        super(IterableDataset, self).__init__()
        self.dataset = dataset
        self.buffer_size = buffer_size

    def buffer_shuffle(self, data_iter):
        """Yield randomly selected samples while incrementally refilling a buffer."""
        buffer = []
        try:
            # Prime the buffer or stop early when the source is smaller than it.
            for _ in range(self.buffer_size):
                buffer.append(next(data_iter))
        except StopIteration:
            pass

        while buffer:
            # Randomize the current window before choosing the next sample.
            random.shuffle(buffer)
            yield buffer.pop()

            try:
                # Refill the window so memory use remains bounded.
                buffer.append(next(data_iter))
            except StopIteration:
                pass

    def __iter__(self):
        """Return a buffered, shuffled iterator over the wrapped dataset."""
        data_iter = iter(self.dataset)
        return iter(self.buffer_shuffle(data_iter))
