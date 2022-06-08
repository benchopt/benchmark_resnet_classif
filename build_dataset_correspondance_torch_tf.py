import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors
import tensorflow_datasets as tfds
import torch
import torchvision
import torchvision.transforms as transforms


def find_permutation(X1, X2, tol=1e-5):
    """
    Find the matching permutation between two arrays X1 and X2
    """
    nn = NearestNeighbors(n_neighbors=2, algorithm='brute')
    nn.fit(X1)
    dists, idx = nn.kneighbors(X2, 2)
    primary_idx = idx[:, 0]
    assert (dists[:, 0] < tol).all(), "Unmatched pairs"
    if len(primary_idx) != len(set(primary_idx)):
        # this means there are duplicates in the arrays
        count = Counter(primary_idx)
        assert max(count.values()) == 2, "More than 3 points are the same"
        duplicate_indices_1 = [
            i for i, v in count.items() if v > 1
        ]
        duplicate_indices_2 = np.where(np.isin(
            primary_idx,
            duplicate_indices_1,
        ))[0]
        for i_2 in sorted(duplicate_indices_2):
            i_1 = primary_idx[i_2]
            potential_i_2 = sorted(np.where(primary_idx == i_1)[0])
            if i_2 > potential_i_2[0]:
                potential_i_1 = idx[i_2]
                alternate_i_1 = [
                    i for i in potential_i_1 if i != i_1
                ][0]
                primary_idx[i_2] = alternate_i_1
    assert len(primary_idx) == len(set(primary_idx)), "Points assigned twice"
    return primary_idx


def find_permutation_per_labels(X1, y1, X2, y2, tol=1e-4):
    labels = set(y1)
    assert labels == set(y2), "inconsistent labels"
    n, _ = X1.shape
    assert X2.shape == X1.shape
    assert len(y1) == n
    assert len(y2) == n
    permutation = np.arange(n)
    for label in labels:
        where1 = np.where(y1 == label)[0]
        where2 = np.where(y2 == label)[0]
        label_perm = find_permutation(X1[where1], X2[where2], tol=tol)
        permutation[where2] = where1[label_perm]

    assert len(set(permutation)) == n
    return permutation


def get_numpy_from_torch(dataset_name, train=True):
    if dataset_name == "cifar10":
        dataset_klass = torchvision.datasets.CIFAR10
        split_kwarg = dict(train=train)
    elif dataset_name == "cifar100":
        dataset_klass = torchvision.datasets.CIFAR100
        split_kwarg = dict(train=train)
    elif dataset_name == "mnist":
        dataset_klass = torchvision.datasets.MNIST
        split_kwarg = dict(train=train)
    elif dataset_name == "svhn_cropped":
        dataset_klass = torchvision.datasets.SVHN
        split_kwarg = dict(split="train" if train else "test")
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
    dataset = dataset_klass(
            root="./data",
            download=True,
            transform=transforms.ToTensor(),
            **split_kwarg,
        )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=len(dataset), shuffle=False
    )
    X_torch, y_torch = next(iter(loader))
    X = np.array(X_torch).transpose(0, 2, 3, 1).reshape(len(dataset), -1)
    y = np.array(y_torch)
    return X, y


def get_numpy_from_tf(dataset_name, train=True):
    split = "train" if train else "test"
    ds = tfds.load(dataset_name, split=split, as_supervised=True)
    X, y = ds.batch(len(ds)).as_numpy_iterator().next()
    X = X.reshape(len(ds), -1) / 255
    return X, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force_registration")
    args = parser.parse_args()
    datasets = ["cifar10", "mnist", "svhn_cropped", "cifar100"]
    for dataset in datasets:
        for train in [True, False]:
            print(f"Registration for {dataset}, train = {train}")
            trainstr = "train" if train else "test"
            registration_dir = Path("./torch_tf_datasets_registrations/")
            filepath = registration_dir / f"{dataset}_{trainstr}.npy"
            if args.force_registration or not filepath.exists():
                X_torch, y_torch = get_numpy_from_torch(dataset, train)
                X_tf, y_tf = get_numpy_from_tf(dataset, train)
                perm = find_permutation_per_labels(
                    X_torch,
                    y_torch,
                    X_tf,
                    y_tf,
                )
                np.save(filepath, perm)
