from typing import *
import gc, os
from abc import ABC, abstractmethod
from synthesizrr.base.framework.mixins import InputOutputDataMixin
from synthesizrr.base.constants import DataLayout, MLType, DataPosition
from synthesizrr.base.util import optional_dependency, safe_validate_arguments, MappedParameters, get_default
from pydantic import validator, root_validator, conint

with optional_dependency('torch'):
    import torch
    from torch.utils.data import IterableDataset as TorchIterableDataset, DataLoader as TorchDataLoader


    class PyTorchTaskDataDataset(TorchIterableDataset):
        """
        PyTorch has two kinds of datasets: map-style (torch.utils.data.Dataset) and iterable-style
        (torch.utils.data.IterableDataset).
        1. Map-style datasets require random access to the i'th element in the dataset: this is done by expecting the
        subclass of torch.utils.data.Dataset to overload __getitem__(self, i). This style of dataset is only
        practical when your dataset can fit in memory, e.g. a list or Pandas DataFrame. With map-style datasets, PyTorch
        must also be able to cheaply get the number of rows in your dataset...this data is exposed by overriding
        __len__(self). When huge data sits on disk or cloud storage, even calculating the number of rows can be
        expensive.
        2. Iterable-style datasets are essential for datasets which do not fit in memory, and instead must be streamed
        from disk or cloud storage like S3. In an IterableDataset, we must only overload __iter__, which will be called
        to get elements or batches of data.
        In DistributedDataParallel training, it is expected that your model fits on a single GPU, but your datasets are
        huge and don't fit in memory (if both your model and dataset fits in memory, you should not be using iterable
        datasets, since they will always be slower than map-style datasets).
        So IterableDatasets become essential in distributed training for very large datasets.

        First, let's talk about how DistributedDataParallel workers will consume a stream of data:
        - Imagine we have a large amount of data (N rows) which arrives in a stream of batches.
        - Each batch in the stream has B rows, where B << N (e.g. B=16, N=10_000_000). The final batch in the
         stream will have <= B rows (because it might happen that N%B !=0, so we will have some left-over
         rows in the last batch).
        - Now, imagine we have S workers (e.g. one worker corresponds to one GPU in DistributedDataParallel).
        Each worker will be assigned one batch of B rows, in a round-robin fashion. Since data arrives as a
        stream, this round-robin allocation is done by assigning the i'th batch to the s'th worker when the
        condition (i % S == s) is satisfied (here, 0<=i<=math.ceil(N/B) and 0<=s<=S-1). Note that there is
        nothing special happening till now...(i % S == s) is the standard way of doing round-robin
        assignment. This is illustrated below (assuming S=4):
          █   █   █   █   █   █   █   █   █   █   ... the stream is very long since N>>B (but not infinite)
          ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓
          s0  s1  s2  s3  s0  s1  s2  s3  s0  s1  ... allocate batch "i" to worker "s", when (i%S == s)
          █ = one batch of B rows (e.g. B=16).

        In reality, we don't have a centralized stream of data. Instead, each worker independently streams
        data from disk/S3, and selects each s'th batch from the stream using (i%S == s) condition.
        Concretely, in PyTorch Iterable-style datasets, each worker process gets a replica of the
        IterableDataset  object. The assumption here is that these Dataset objects are lightweight and only
        contain metadata (e.g. the path to the data-folder on disk/cloud, rather than the data itself). A
        worker process can fetch elements/batches of data by calling __iter__ on its corresponding replica
        of IterableDataset (not: you don't need to write code to do this youtself...pass the IterableDataset
        replica into a dataloader, and PyTorch will call __iter__ for you when you loop over the dataloader).
        However, calling __iter__ naively in such a multi-worker fashion has an issue: since each
        IterableDataset replica has no knowledge of the others, we end up loading the whole dataset
        "S" times from disk/S3 for each epoch, even though each worker only uses one of every "S" batches.
        So, naively, we unnecessarily load "S-1" copies of the dataset, distributed across workers. This
        is illustrated below: "▒" show batches of data were loaded but unused by the worker.
          s0: █   ▒   ▒   ▒   █   ▒   ▒   ▒   █   ▒   ...
          s1: ▒   █   ▒   ▒   ▒   █   ▒   ▒   ▒   █   ...
          s2: ▒   ▒   █   ▒   ▒   ▒   █   ▒   ▒   ▒   ...
          s3: ▒   ▒   ▒   █   ▒   ▒   ▒   █   ▒   ▒   ...
          █ = loaded from disk/S3 & used by worker, ▒ = loaded from disk/S3, but unused by worker
        A more efficient way to do this is to "shard" the dataset i.e. partition the dataset into, say,
        K*S files, each file having an equal number of rows. This is done as a preprocessing step, before
        we begin loading data onto any of the workers. For example, imagine we have S=4 and K=10, and thus
        have sharded our dataset into 10*4=40 files on disk/S3. Then the s'th worker has to load data only
        from files where (file_i % S == s).
        So, worker s0 will load all the batches from the dataset-shards where file_i=[0, 4, 8, 12, ...],
        worker s1 will load all batches from dataset-shards where file_i=[1, 5, 9, 13, ...], etc.
        Generalizing, the worker with rank "s" will load all batches from dataset-shards where
        (file_i % S == s). Here, we only load the whole dataset once, avoiding a lot of unnecessary
        data-transfer from disk/cloud. This is illustrated below.
          s0:  █   █   █   █   █   █   █   █   █   █   ... <-- only load from shards where (shard_rank % S == 0)
          s1:  █   █   █   █   █   █   █   █   █   █   ... <-- only load from shards where (shard_rank % S == 1)
          s2:  █   █   █   █   █   █   █   █   █   █   ... <-- only load from shards where (shard_rank % S == 2)
          s3:  █   █   █   █   █   █   █   █   █   █   ... <-- only load from shards where (shard_rank % S == 3)
        step:  0   1   2   3   4   5   6   7   8   9   ...
          █ = loaded from disk/S3 & used by worker.
        When doing DistributedDataParallel, each "step" of training computes one column of S*B rows (refer
        to the "step" axis in the illustration above). Concretely, each worker gets a batch of "B" rows
        (loaded efficiently from their corresponding shard as described above), and calculates their
        gradient update. Then, we synchronize & apply gradient updates on all workers. Another way to think
        about it is that each gradient step is done using a "batch" of B*S rows (in DistributedDataParallel,
        "B" is typically called the "micro-batch" size).

        When sharding to load efficiently, we run into an issue when (N % S*B) != 0, i.e. the last "column"
        does not have enough data. This almost always happens, since it is unlikely that "N" (the number of
        rows in your training dataset) is perfectly divisible by S*B.
        This issue is illustrated below: we will not be able to run the last step of training using
        DistributedDataParallel, since we don't have enough data. Remember, data can only be streamed in
        batches of <= B rows.
          s0:  █   █   █   █   █   █   █   █   █   █    ... ... █
          s1:  █   █   █   █   █   █   █   █   █   █    ... ... ▀  <- last batch in incomplete as N%B != 0
          s2:  █   █   █   █   █   █   █   █   █   █    ... ... ░
          s3:  █   █   █   █   █   █   █   █   █   █    ... ... ░
        step:  0   1   2   3   4   5   6   7   8   9    ... ... last
                                                                ⮑ last column in incomplete as N%(S*B) != 0
          █ = complete batch of size B, ▀ = incomplete batch of size <= B, ░ = missing batch (no data)
        In this case, it's best to skip the last step/column altogether. This is usually okay because
        N >> S*B, e.g. with N=1,000,000 rows, B=16 batch-size & S=24 workers, we will run a total of
        2,604 gradient-update steps before we reach the last (incomplete) column. So long as we shuffle
        our dataset each epoch, skipping the last (incomplete) column of data has a negligible effect
        on the training procedure. The drop_last flag controls whether we skip the last (incomplete) column
        of data.
        Note: we can detect we are in the last column when (N - steps*S*B) is strictly less than S*B.
        This requires that we know our train dataset length "N" in advance. In the case of our codebase,
        this should not be an overhead, because we do a pass over the train dataset beforehand to calculate
        training statistics.
        """

        def __init__(
                self,
                dataset: InputOutputDataMixin,
                **kwargs,
        ):
            kwargs.pop('device', None)
            self.dataset: InputOutputDataMixin = dataset
            self.batching_kwargs: Dict[str, Any] = kwargs
            # print(f'pid={os.getpid()} corresponds to rank={self.rank}, num_workers={self.num_workers}\n')

        def __iter__(self):
            ## Ref: https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                ## We are in a worker process.
                ## Refs:
                ## - https://medium.com/speechmatics/how-to-build-a-streaming-dataloader-with-pytorch-a66dd891d9dd
                ## print(f'(PID: {os.getpid()}) worker_info: {worker_info}')
                ## TODO: update this comment to explain why we are NOT doing multi-worker processing using PyTorch.
                return self._create_generator()
            else:
                ## Load data in the "main" process (i.e. where the dataloading for-loop is running)
                return self._create_generator()

        def _create_generator(self) -> Generator[InputOutputDataMixin, None, None]:
            step: int = 0
            for i, batch in enumerate(self.dataset.iter(**self.batching_kwargs)):
                step += 1
                yield batch

        def dataloader(self) -> TorchDataLoader:
            dataloader: TorchDataLoader = TorchDataLoader(
                ## num_workers=0 loads data in the main-process itself.
                ## Behind the scenes, we use InputOutputDataMixin.read_batches() to do multi-process dataloading,
                ## rather than relying on PyTorch's multi-process dataloading logic.
                ## When running in a non-distributed setting, this means that we do multi-process dataloading to feed
                ## data batches to the main-process (which might, e.g. the kernel of the Juptyter notebook).
                ## When doing distribute training using HF Accelerate or torch.distributed, suppose we have `N` copies
                ## of the model; we will then have `N` main-processes (whereas in the non-distributed case, we had only
                ## one main-process: the Jupyter kernel). Thus in this distributed setting, each GPU's main-process will
                ## perform multi-process dataloading: i.e. we have `N` worker-pools, each with `K` worker processes, for
                ## a total of N*K workers. Depending on your machine hardware, N*K might be a good fraction of your
                ## total CPUs, so ideally set K to be low (say, K=4 or K=8).
                self,
                num_workers=0,
                ## This tells the torch dataloader that we will do batching ourselves i.e. using
                ## InputOutputDataMixin.read_batches().
                batch_size=None,
            )
            return dataloader
