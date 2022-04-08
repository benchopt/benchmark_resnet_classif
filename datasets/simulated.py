from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    import tensorflow as tf
    import torch
    from torch.utils.data import TensorDataset

    MultiFrameworkDataset = import_ctx.import_from(
        'multi_frameworks_dataset',
        'MultiFrameworkDataset',
    )


class Dataset(MultiFrameworkDataset):

    name = "Simulated"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        'n_samples, img_size': [
            (128, 32),
        ],
        'framework': ['pytorch', 'tensorflow'],
    }

    ds_description = dict(
        n_classes=2,
    )

    def __init__(
        self,
        n_samples=10,
        img_size=50,
        train_frac=0.8,
        framework='pytorch',
        random_state=27,
    ):
        # Store the parameters of the dataset
        self.n_samples = n_samples
        self.img_size = img_size
        self.train_frac = train_frac
        self.framework = framework
        self.random_state = random_state
        self.rng = np.random.default_rng(self.random_state)

    def _get_data(self):
        n_train = int(self.n_samples * self.train_frac)
        self.ds_description['n_samples_train'] = n_train
        self.ds_description['n_samples_test'] = self.n_samples - n_train
        self.ds_description['image_width'] = self.img_size
        # inputs are channel first
        inps = self.rng.normal(
            size=(self.n_samples, 3, self.img_size, self.img_size,),
        ).astype(np.float32)
        tgts = self.rng.integers(0, 2, (self.n_samples,)).astype(np.int32)
        inps_train, inps_test = inps[:n_train], inps[n_train:]
        tgts_train, tgts_test = tgts[:n_train], tgts[n_train:]
        return inps_train, inps_test, tgts_train, tgts_test

    def get_torch_data(self):
        inps_train, inps_test, tgts_train, tgts_test = self._get_data()
        dataset = TensorDataset(
            torch.Tensor(inps_train),
            torch.Tensor(tgts_train),
        )
        test_dataset = TensorDataset(
            torch.Tensor(inps_test),
            torch.Tensor(tgts_test),
        )

        data = dict(
            dataset=dataset,
            test_dataset=test_dataset,
            **self.ds_description,
        )

        return 'object', data

    def get_tf_data(self):
        inps_train, inps_test, tgts_train, tgts_test = self._get_data()
        dataset = tf.data.Dataset.from_tensor_slices((inps_train, tgts_train))
        test_dataset = tf.data.Dataset.from_tensor_slices(
            (inps_test, tgts_test),
        )

        data = dict(
            dataset=dataset,
            test_dataset=test_dataset,
            **self.ds_description,
        )

        return 'object', data
