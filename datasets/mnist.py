import numpy as np

from benchopt import BaseDataset
import torchvision
from torchvision import transforms
import torchvision.datasets as datasets


class Dataset(BaseDataset):

    name = "MNIST"

    parameters = {
    }

    def __init__(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        self.mnist_trainset = datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transform,
        )
        self.n_features = 28**2


    def get_data(self):
        data = dict(dataset=self.mnist_trainset)

        return self.n_features, data
