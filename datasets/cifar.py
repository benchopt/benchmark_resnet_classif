import numpy as np

from benchopt import BaseDataset
import torchvision
from torchvision import transforms
import torchvision.datasets as datasets


class Dataset(BaseDataset):

    name = "CIFAR"

    parameters = {
    }

    def __init__(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.cifar_trainset = datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=transform,
        )
        self.n_features = (32**2)*3


    def get_data(self):
        data = dict(dataset=self.cifar_trainset)

        return self.n_features, data
