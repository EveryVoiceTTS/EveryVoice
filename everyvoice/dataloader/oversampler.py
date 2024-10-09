from typing import Iterator, Sized

import torch
from torch.utils.data.sampler import Sampler, SequentialSampler


class BatchOversampler(Sampler[list[int]]):
    r"""Samples elements sequentially, always in the same order. Completes the last incomplete batch with random samples from other batches.

    Args:
        data_source (Dataset): dataset to sample from
        batch_size (int): number of items in a batch
    """

    def __init__(self, data_source: Sized, batch_size: int) -> None:
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError(
                f"batch_size should be a positive integer value, but got batch_size={batch_size}"
            )
        self.batch_size = batch_size
        self.n = len(data_source)
        self.n_full_batches = self.n // self.batch_size
        self.remaining_samples = self.n % self.batch_size
        self.sampler = SequentialSampler(data_source)

    def __iter__(self) -> Iterator[list[int]]:
        batch = [0] * self.batch_size
        idx_in_batch = 0
        for idx in self.sampler:
            batch[idx_in_batch] = idx
            idx_in_batch += 1
            if idx_in_batch == self.batch_size:
                yield batch
                idx_in_batch = 0
                batch = [0] * self.batch_size
        if idx_in_batch > 0:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
            oversampler = map(
                int,
                torch.randperm(
                    self.n_full_batches * self.batch_size, generator=generator
                )[: self.batch_size - self.remaining_samples].numpy(),
            )
            for idx in oversampler:
                batch[idx_in_batch] = idx
                idx_in_batch += 1
            yield batch

    def __len__(self) -> int:
        if self.remaining_samples:
            return self.n_full_batches + 1
        else:
            return self.n_full_batches
