import math
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset
from torchvision.datasets import CIFAR10, CIFAR100


class WorkerInitFn:
    def __init__(self, base_seed, rank):
        self.base_seed = base_seed
        self.rank = rank
    def __call__(self, worker_id):
        seed = (self.base_seed + self.rank * 1000 + worker_id) % (2**32)
        np.random.seed(seed)
        import random
        random.seed(seed)
        torch.manual_seed(seed)

def get_worker_init_fn(base_seed, rank):
    return WorkerInitFn(base_seed, rank)

class ProtectedCIFAR:
    def __init__(self, root, dataset_name, mode="dev", split_info=None):
        self.root = root
        self.dataset_name = dataset_name.lower()
        self.mode = mode
        self.split_info = split_info

        if self.dataset_name == "cifar10":
            self.dataset_class = CIFAR10
        elif self.dataset_name == "cifar100":
            self.dataset_class = CIFAR100
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    def _get_official_train(self, download=True):
        return self.dataset_class(root=self.root, train=True, download=download, transform=None)
        
    def _get_official_test(self, download=True):
        if self.mode == "dev":
            raise RuntimeError("Cannot load official test dataset in dev mode.")
        if self.mode == "final":
            if not os.path.exists(os.path.join("experiments", "FREEZE.md")):
                raise RuntimeError("Test split cannot be loaded until FREEZE.md exists in final mode.")
        return self.dataset_class(root=self.root, train=False, download=download, transform=None)

    def get_train_dataset(self, transform, download=True):
        ds = self._get_official_train(download=download)
        if self.mode == "dev":
            if self.split_info is None:
                raise ValueError("split_info must be provided in dev mode.")
            ds = Subset(ds, self.split_info["train_indices"])
        
        if isinstance(ds, Subset):
            return TransformedSubset(ds.dataset, ds.indices, transform)
        ds.transform = transform
        return ds

    def get_val_dataset(self, transform, download=True):
        if self.mode == "dev":
            if self.split_info is None:
                raise ValueError("split_info must be provided in dev mode.")
            ds = self._get_official_train(download=download)
            return TransformedSubset(ds, self.split_info["val_indices"], transform)
        else:
            # In final mode, validation is not strictly defined, or we could return None.
            # Usually test set is used for final eval.
            raise RuntimeError("Val dataset is not available in final mode, use test dataset after freezing.")

    def get_test_dataset(self, transform, download=True):
        ds = self._get_official_test(download=download)
        ds.transform = transform
        return ds

class TransformedSubset(Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        img, target = self.dataset[self.indices[idx]]
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.indices)

class ExactDistributedSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False, seed=0, drop_last=False):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed, drop_last=drop_last)
        if not self.drop_last:
            self.total_size = len(self.dataset)
            # The length per replica varies; we don't pad.
            self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
            # Actually, we shouldn't rely on self.num_samples for exact size in len(), it can be different per rank.
            # rank i gets: length // num_replicas + (1 if length % num_replicas > i else 0)
            
    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
            
        if self.drop_last:
            indices = indices[:(len(indices) // self.num_replicas) * self.num_replicas]
            
        indices = indices[self.rank:len(indices):self.num_replicas]
        return iter(indices)
        
    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.num_replicas
        else:
            return math.ceil((len(self.dataset) - self.rank) / self.num_replicas)

def create_dataloaders(
    dataset_obj, train_transform, eval_transform,
    global_batch_size, num_workers, distributed=False, seed=2026, world_size=1
):
    if global_batch_size % world_size != 0:
        raise ValueError(f"global_batch_size {global_batch_size} must be divisible by world_size {world_size}")
    batch_size = global_batch_size // world_size

    rank = int(os.environ.get("RANK", 0)) if distributed else 0

    train_ds = dataset_obj.get_train_dataset(train_transform)
    
    g = torch.Generator()
    g.manual_seed(seed)
    
    if distributed:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, drop_last=True, shuffle=True)
    else:
        train_sampler = None
        
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=(train_sampler is None), 
        sampler=train_sampler,
        num_workers=num_workers,
        worker_init_fn=get_worker_init_fn(seed, rank),
        generator=g,
        drop_last=True
    )
    
    if dataset_obj.mode == "dev":
        val_ds = dataset_obj.get_val_dataset(eval_transform)
        if distributed:
            val_sampler = ExactDistributedSampler(
                val_ds, num_replicas=world_size, rank=rank, drop_last=False, shuffle=False
            )
        else:
            val_sampler = None
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=num_workers
        )
        return train_loader, val_loader
    else:
        return train_loader, None
