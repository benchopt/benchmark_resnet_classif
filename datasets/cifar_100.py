from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import torchvision.datasets as datasets

    MultiFrameworkDataset = import_ctx.import_from(
        'multi_frameworks_dataset',
        'MultiFrameworkDataset',
    )


class Dataset(MultiFrameworkDataset):

    name = "CIFAR-100"

    # from
    # https://jovian.ai/kumar-shailesh1597/cifar100-resnet18
    normalization_mean = (0.5071, 0.4867, 0.4408)
    normalization_std = (0.2675, 0.2565, 0.2761)

    ds_description = dict(
        n_samples_train=40_000,
        n_samples_val=10_000,
        n_samples_test=10_000,
        image_width=32,
        n_classes=100,
        symmetry='horizontal',
    )

    torch_ds_klass = datasets.CIFAR100

    tf_ds_name = 'cifar100'
