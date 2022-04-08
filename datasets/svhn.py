from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    from torchvision import transforms
    import torchvision.datasets as datasets


class Dataset(BaseDataset):

    name = "SVHN"

    def get_data(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            # from
            # https://deepobs.readthedocs.io/en/v1.2.0-beta0_a/_modules/deepobs/pytorch/datasets/svhn.html
            transforms.Normalize(
                (0.4376821, 0.4437697, 0.47280442),
                (0.19803012, 0.20101562, 0.19703614),
            ),
        ])
        svhn_trainset = datasets.SVHN(
            root='./data',
            split='train',
            download=True,
            transform=transform,
        )
        svhn_testset = datasets.SVHN(
            root='./data',
            split='train',
            download=True,
            transform=transform,
        )

        return 'object', dict(dataset=svhn_trainset, test_dataset=svhn_testset)
