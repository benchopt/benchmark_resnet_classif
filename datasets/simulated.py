from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    import torch
    from torch.utils.data import TensorDataset


class Dataset(BaseDataset):

    name = "Simulated"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        'n_samples, img_size': [
            (128, 32),
        ]
    }

    def __init__(self, n_samples=10, img_size=50, train_frac=0.8,
                 random_state=27):
        # Store the parameters of the dataset
        self.n_samples = n_samples
        self.img_size = img_size
        self.train_frac = train_frac
        self.random_state = random_state

    def get_data(self):
        n_train = int(self.n_samples * self.train_frac)
        inps = torch.randn(self.n_samples, 3, self.img_size, self.img_size,
                           dtype=torch.float32)
        tgts = torch.randint(0, 2, (self.n_samples,))
        dataset = TensorDataset(inps[:n_train], tgts[:n_train])
        test_dataset = TensorDataset(inps[n_train:], tgts[n_train:])

        data = dict(dataset=dataset, test_dataset=test_dataset)

        return 'object', data
