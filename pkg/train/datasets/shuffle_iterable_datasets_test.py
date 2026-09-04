"""Manual smoke test for buffered iterable-dataset shuffling."""

from torch.utils.data import DataLoader

from pkg.train.datasets.shuffle_iterable_datasets import ShuffledIterableDataset

if __name__ == "__main__":
    data = list(range(50))

    buffer_size = 10

    # Wrap the source sequence in a buffered streaming dataset.
    dataset = ShuffledIterableDataset(data, buffer_size)

    # Batch the shuffled samples with PyTorch's standard loader.
    data_loader = DataLoader(dataset, batch_size=4)

    # Print every batch for a quick visual distribution check.
    batch_num = 0
    for batch in data_loader:
        batch_num += 1
        print(batch_num, batch)
