from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import torchvision.datasets as datasets

    MultiFrameworkDataset = import_ctx.import_from(
        'multi_frameworks_dataset',
        'MultiFrameworkDataset',
    )


class Dataset(MultiFrameworkDataset):

    name = "SVHN"

    # from
    # https://deepobs.readthedocs.io/en/v1.2.0-beta0_a/_modules/deepobs/pytorch/datasets/svhn.html
    normalization_mean = (0.4376821, 0.4437697, 0.47280442)
    normalization_std = (0.19803012, 0.20101562, 0.19703614)

    ds_description = dict(
        n_samples_train=58_605,
        n_samples_val=14_652,
        n_samples_test=26_032,
        image_width=32,
        n_classes=10,
    )

    torch_ds_klass = datasets.SVHN
    torch_split_kwarg = 'split'

    tf_ds_name = 'svhn_cropped'
    # XXX: problem with the tfds, it downloads the full svhn dataset
    # including the extra bit which is super heavy
    # we might be able to limit the download with
    # https://www.tensorflow.org/datasets/api_docs/python/tfds/download/DownloadConfig
    # `max_examples_per_split`
