# fed/datasets.py

import numpy as np
from typing import Dict, List
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_cifar100_datasets(data_dir: str = "./data"):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=transform_test
    )
    return train_dataset, test_dataset


def iid_partition(train_dataset, num_clients: int) -> Dict[int, List[int]]:
    """
    Simple IID split: shuffle indices and divide equally among clients.
    """
    n = len(train_dataset)
    indices = np.random.permutation(n)
    shard_size = n // num_clients
    client_dict = {}

    for cid in range(num_clients):
        start = cid * shard_size
        end = n if cid == num_clients - 1 else (cid + 1) * shard_size
        client_dict[cid] = indices[start:end].tolist()

    return client_dict


def dirichlet_partition(
    train_dataset,
    num_clients: int,
    alpha: float = 0.5,
    num_classes: int = 100,
    min_size: int = 10,
) -> Dict[int, List[int]]:
    """
    Non-IID partition using a Dirichlet distribution over label proportions.
    """
    # Get labels array
    if hasattr(train_dataset, "targets"):
        targets = np.array(train_dataset.targets)
    elif hasattr(train_dataset, "labels"):
        targets = np.array(train_dataset.labels)
    else:
        raise ValueError("Dataset must have 'targets' or 'labels' attribute for labels.")

    client_indices = {cid: [] for cid in range(num_clients)}

    while True:
        client_indices = {cid: [] for cid in range(num_clients)}

        for c in range(num_classes):
            idxs_c = np.where(targets == c)[0]
            np.random.shuffle(idxs_c)

            proportions = np.random.dirichlet(alpha * np.ones(num_clients))
            counts = (proportions * len(idxs_c)).astype(int)

            diff = len(idxs_c) - np.sum(counts)
            for _ in range(abs(diff)):
                j = np.random.randint(0, num_clients)
                counts[j] += 1 if diff > 0 else -1

            start = 0
            for cid in range(num_clients):
                cnt = counts[cid]
                if cnt > 0:
                    client_indices[cid].extend(
                        idxs_c[start : start + cnt].tolist()
                    )
                    start += cnt

        sizes = [len(v) for v in client_indices.values()]
        if min(sizes) >= min_size:
            break
        else:
            print("Retrying Dirichlet partition, client sizes:", sizes)

    for cid in range(num_clients):
        np.random.shuffle(client_indices[cid])

    return client_indices


def get_client_loaders(train_dataset, client_indices: Dict[int, List[int]], batch_size: int = 64):
    client_loaders = {}
    for cid, idxs in client_indices.items():
        subset = Subset(train_dataset, idxs)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        client_loaders[cid] = loader
    return client_loaders


def get_test_loader(test_dataset, batch_size: int = 128):
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


#SUBSET
from torch.utils.data import Subset

def get_loader_from_indices(dataset, indices, batch_size=64):
    """
    Build a DataLoader from a subset of dataset indices.
    """
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
    return loader
