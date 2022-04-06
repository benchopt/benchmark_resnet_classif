
from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    from torch.utils.data import Subset
    from torchvision import transforms
    import torchvision.datasets as datasets


def grayscale_to_rbg(image):
    return image.repeat(3, 1, 1)


class Dataset(BaseDataset):

    name = "MNIST"
    parameters = {
        'debug': [True]
    }

    def get_data(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            # from
            # https://stackoverflow.com/a/67233938/4332585
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(grayscale_to_rbg),
        ])
        mnist_trainset = datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transform,
        )
        if self.debug:
            mnist_trainset = Subset(mnist_trainset, range(1000))

        return 'object', dict(dataset=mnist_trainset)
