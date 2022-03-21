import numpy as np

from benchopt import BaseDataset
import torchvision
import torchvision.datasets as datasets


class Dataset(BaseDataset):

    name = "MNIST"

    parameters = {
    }

    def __init__(self):
        self.mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
        self.X = self.mnist_trainset.data.numpy()
        self.y = self.mnist_trainset.targets.numpy()
        self.n_features = self.X.shape[1]*self.X.shape[2]


    def get_data(self):
        data = dict(X=self.X, y=self.y)

        return self.n_features, data
