from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import tensorflow as tf
    import torchvision.datasets as datasets
    from torchvision import transforms

    MultiFrameworkDataset = import_ctx.import_from(
        'multi_frameworks_dataset',
        'MultiFrameworkDataset',
    )


class Dataset(MultiFrameworkDataset):

    name = "Imagenet"

    parameters = {
        'framework': ['pytorch', 'lightning', 'tensorflow'],
        'random_state': [42],
        'with_validation': [False],
    }

    # from
    # https://github.com/pytorch/examples/blob/main/imagenet/main.py#L211-L212
    normalization_mean = (0.485, 0.456, 0.406)
    normalization_std = (0.229, 0.224, 0.225)

    ds_description = dict(
        n_samples_train=1_281_167,
        n_samples_val=0,
        n_samples_test=50_000,
        # as in
        # https://github.com/pytorch/examples/blob/main/imagenet/main.py#L217
        image_width=224,
        n_classes=1000,
        symmetry='horizontal',
    )

    torch_ds_klass = datasets.ImageNet
    torch_split_kwarg = 'split'
    torch_dl = False
    extra_torch_test_transforms = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]

    tf_ds_name = 'imagenet2012'
    tf_splits = ['validation', 'train']

    def get_torch_splits(self):
        return ["train", "val"]

    def tf_test_image_processing(self, image):
        normalization = self.get_tf_preprocessing_step()
        image = normalization(image)
        # resize to 256x256 and center crop to 224x224
        image = tf.image.resize(image, [256, 256])
        image = tf.image.central_crop(image, central_fraction=0.875)
        return image
