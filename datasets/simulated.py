from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    import tensorflow as tf
    import torch
    from torch.utils.data import TensorDataset
    from torchvision import transforms

    AugmentedDataset = import_ctx.import_from(
        'torch_helper', 'AugmentedDataset'
    )
    TFDatasetCapsule = import_ctx.import_from(
        'tf_helper', 'TFDatasetCapsule'
    )
    MultiFrameworkDataset = import_ctx.import_from(
        'multi_frameworks_dataset',
        'MultiFrameworkDataset',
    )

    MultiFrameworkDataset = import_ctx.import_from(
        'multi_frameworks_dataset',
        'MultiFrameworkDataset',
    )


def make_channels_last(images):
    return np.transpose(images, (0, 2, 3, 1))


def make_channels_last(images):
    return np.transpose(images, (0, 2, 3, 1))


class Dataset(MultiFrameworkDataset):

    name = "Simulated"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        'n_samples, img_size': [
            (128, 32),
        ],
        # WARNING: this order is very important
        # as tensorflow takes all the memory and doesn't have a mechanism to
        # release it
        'framework': ['pytorch', 'lightning', 'tensorflow'],
    }

    # This makes sure that for each solver, we have one simulated dataset that
    # will be compatible in the test_solver.
    test_parameters = {
        'framework': ['pytorch', 'lightning', 'tensorflow'],
    }

    def __init__(
        self,
        n_samples=10,
        img_size=32,
        n_classes=2,
        train_frac=0.8,
        framework='lightning',
        random_state=27,
    ):
        # Store the parameters of the dataset
        self.n_samples = n_samples
        self.img_size = img_size
        self.n_classes = n_classes
        self.train_frac = train_frac
        self.framework = framework
        self.random_state = random_state
        self.rng = np.random.default_rng(self.random_state)

    def get_np_data(self):
        n_train = int(self.n_samples * self.train_frac)

        # Get data description
        self.ds_description = dict(
            n_samples_train=n_train,
            n_samples_test=self.n_samples - n_train,
            image_width=self.img_size,
            n_classes=self.n_classes
        )

        # inputs are channel first
        inps = self.rng.normal(
            size=(self.n_samples, 3, self.img_size, self.img_size,),
        ).astype(np.float32)
        tgts = self.rng.integers(
            0, self.n_classes, (self.n_samples,)
        ).astype(np.int64)
        inps_train, inps_test = inps[:n_train], inps[n_train:]
        tgts_train, tgts_test = tgts[:n_train], tgts[n_train:]
        return inps_train, inps_test, tgts_train, tgts_test

    def get_torch_data(self):
        inps_train, inps_test, tgts_train, tgts_test = self.get_np_data()
        dataset = TensorDataset(
            torch.tensor(inps_train), torch.tensor(tgts_train),
        )
        test_dataset = TensorDataset(
            torch.tensor(inps_test), torch.tensor(tgts_test),
        )
        return dataset, test_dataset

    def get_tf_data(self):
        inps_train, inps_test, tgts_train, tgts_test = self.get_np_data()
        dataset = tf.data.Dataset.from_tensor_slices(
            (make_channels_last(inps_train), tgts_train),
        )
        test_dataset = tf.data.Dataset.from_tensor_slices(
            (make_channels_last(inps_test), tgts_test),
        )
        return dataset, test_dataset

    def get_data(self):
        """Switch to select the data from the right framework."""
        if self.framework in ['lightning', 'pytorch']:
            dataset, test_dataset = self.get_torch_data()
            normalization = None
        elif self.framework == 'tensorflow':
            dataset, test_dataset = self.get_tf_data()

            def normalization(x):
                return x
        else:
            raise ValueError(f"Framework not supported {self.framework}")

        data = dict(
            dataset=dataset,
            test_dataset=test_dataset,
            framework=self.framework,
            normalization=normalization,
            **self.ds_description,
        )

        return 'object', data
