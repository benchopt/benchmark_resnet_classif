import numpy as np

from benchopt import BaseDataset
import torchvision
from torchvision import transforms
import torchvision.datasets as datasets


class Dataset(BaseDataset):

    name = "SVHN"

    parameters = {
    }

    def __init__(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.svhn_trainset = datasets.SVHN(
            root='./data',
            split='train',
            download=True,
            transform=transform,
        )
        self.n_features = (32**2)*3


    def get_data(self):
        data = dict(dataset=self.svhn_trainset)

        return self.n_features, data
