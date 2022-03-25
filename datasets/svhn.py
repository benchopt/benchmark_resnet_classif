from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    from torchvision import transforms
    import torchvision.datasets as datasets


class Dataset(BaseDataset):

    name = "SVHN"

    def get_data(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        svhn_trainset = datasets.SVHN(
            root='./data',
            split='train',
            download=True,
            transform=transform,
        )

        return 'object', dict(dataset=svhn_trainset)
