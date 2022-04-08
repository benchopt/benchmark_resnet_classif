from abc import ABC, abstractmethod

from benchopt import BaseDataset


class MultiFrameworkDataset(BaseDataset, ABC):
    def __init__(self, framework='pytorch'):
        # Store the parameters of the dataset
        self.framework = framework

    @abstractmethod
    def get_torch_data(self):
        ...

    @abstractmethod
    def get_tf_data(self):
        ...

    def get_data(self):
        if self.framework == 'pytorch':
            return self.get_torch_data()
        elif self.framework == 'tensorflow':
            return self.get_tf_data()
