import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

import tensorflow_datasets as tfds

from sklearn.neighbors import NearestNeighbors


def find_permutation(X1, X2, tol=1e-5):
    """
    Find the matching permutation between two arrays X1 and X2
    """
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(X1)
    dists, idx = nn.kneighbors(X2)
    idx = idx[:, 0]
    assert (dists < tol).all(), "Unmatched pairs"
    assert len(idx) == len(set(idx)), "points assigned twice !"
    return idx


def find_permutation_labels(X1, y1, X2, y2, tol=1e-4):
    labels = set(y1)
    assert labels == set(y2), "inconsistant labels"
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
    return permutation


def get_numpy_from_torch(dataset_name, train=True):
    if dataset_name == "cifar10":
        dataset = torchvision.datasets.CIFAR10(
            root="./data",
            train=train,
            download=True,
            transform=transforms.ToTensor(),
        )
    elif dataset_name == "mnist":
        dataset = torchvision.datasets.MNIST(
            root="./data",
            train=train,
            download=True,
            transform=transforms.ToTensor(),
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
    datasets = ["cifar10", "mnist"]
    for dataset in datasets:
        for train in [True, False]:
            print(f"Registration for {dataset}, train = {train}")
            X1, y1 = get_numpy_from_torch(dataset, train)
            X2, y2 = get_numpy_from_tf(dataset, train)
            perm = find_permutation_labels(X1, y1, X2, y2)
            trainstr = "train" if train else "test"
            np.save(
                f"./torch_tf_datasets_registrations/{dataset}_{trainstr}.npy",
                perm,
            )
