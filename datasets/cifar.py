from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import torchvision.datasets as datasets

    MultiFrameworkDataset = import_ctx.import_from(
        'multi_frameworks_dataset',
        'MultiFrameworkDataset',
    )


class Dataset(MultiFrameworkDataset):

    name = "CIFAR"

    # from
    # https://github.com/kuangliu/pytorch-cifar/blob/master/main.py#L34
    normalization_mean = (0.4914, 0.4822, 0.4465)
    normalization_std = (0.2023, 0.1994, 0.2010)

    ds_description = dict(
        n_samples_train=50_000,
        n_samples_test=10_000,
        image_width=32,
        n_classes=10,
    )

    torch_ds_klass = datasets.CIFAR10

    tf_ds_name = 'cifar10'
