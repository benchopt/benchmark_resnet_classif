import numpy as np

from benchopt import BaseDataset
import torchvision
from torchvision import transforms
import torchvision.datasets as datasets


def grayscale_to_rbg(image):
    return image.repeat(3, 1, 1)
class Dataset(BaseDataset):

    name = "MNIST"

    parameters = {
    }

    def __init__(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            # because of this transform,
            # the dataset is not picklable, and therefore
            # we cannot use it in benchopt.
            transforms.Lambda(grayscale_to_rbg),
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
