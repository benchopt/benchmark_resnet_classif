from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import torchvision.datasets as datasets

    MultiFrameworkDataset = import_ctx.import_from(
        'multi_frameworks_dataset',
        'MultiFrameworkDataset',
    )


class Dataset(MultiFrameworkDataset):

    name = "Imagenet"

    # from
    # https://github.com/pytorch/examples/blob/main/imagenet/main.py#L211-L212
    normalization_mean = (0.485, 0.456, 0.406)
    normalization_std = (0.229, 0.224, 0.225)

    ds_description = dict(
        n_samples_train=1_281_167 - 50_000,
        n_samples_val=50_000,
        n_samples_test=50_000,
        image_width=None,
        n_classes=1000,
        symmetry='horizontal',
    )

    torch_ds_klass = datasets.ImageNet

    tf_ds_name = 'imagenet2012'
