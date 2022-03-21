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
        self.n_features = 28**2


    def get_data(self):
        data = dict(dataset=self.mnist_trainset)

        return self.n_features, data
