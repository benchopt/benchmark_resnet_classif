from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    import tensorflow as tf
    from torch.utils.data import TensorDataset


class Dataset(BaseDataset):

    name = "Simulated"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        'n_samples, img_size': [
            (128, 32),
        ],
        'framework': ['pytorch', 'tensorflow'],
    }

    def __init__(
        self,
        n_samples=10,
        img_size=50,
        framework='pytorch',
        random_state=27,
    ):
        # Store the parameters of the dataset
        self.n_samples = n_samples
        self.img_size = img_size
        self.framework = framework
        self.random_state = random_state
        self.rng = np.random.default_rng(self.random_state)

    def _get_data(self):
        # inputs are channel first
        inps = self.rng.normal(
            size=(self.n_samples, 3, self.img_size, self.img_size,),
        )
        tgts = self.rng.randint(0, 2, (self.n_samples,))
        return inps, tgts

    def get_data_torch(self):
        inps, tgts = self._get_data()
        dataset = TensorDataset(inps, tgts)

        data = dict(dataset=dataset)

        return 'object', data

    def get_tf_data(self):
        inps, tgts = self._get_data()
        dataset = tf.data.Dataset.from_tensor_slices((inps, tgts))

        data = dict(dataset=dataset)

        return 'object', data

    def get_data(self):
        if self.framework == 'pytorch':
            return self.get_data_torch()
        elif self.framework == 'tensorflow':
            return self.get_tf_data()
