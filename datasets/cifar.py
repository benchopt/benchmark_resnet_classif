from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    from torchvision import transforms
    import torchvision.datasets as datasets


class Dataset(BaseDataset):

    name = "CIFAR"

    def get_data(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            # from
            # https://github.com/kuangliu/pytorch-cifar/blob/master/main.py#L34
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010),
            ),
        ])
        cifar_trainset = datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=transform,
        )
        cifar_testset = datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=transform,
        )

        data = dict(dataset=cifar_trainset, test_dataset=cifar_testset)

        return 'object', data
