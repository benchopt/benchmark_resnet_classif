from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    from torchvision import transforms
    import torchvision.datasets as datasets


class Dataset(BaseDataset):

    name = "CIFAR"

    def get_data(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        cifar_trainset = datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=transform,
        )

        data = dict(dataset=cifar_trainset)

        return 'object', data
