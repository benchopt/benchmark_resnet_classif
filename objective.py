from operator import xor
from benchopt import BaseObjective, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from pytorch_lightning import Trainer
    import tensorflow as tf
    from torch.utils.data import DataLoader
    import torchvision.models as models
    BenchPLModule = import_ctx.import_from('torch_helper', 'BenchPLModule')


class Objective(BaseObjective):
    """Classification objective"""
    name = "ResNet classification fitting"

    install_cmd = 'conda'
    requirements = [
        'pytorch', 'torchvision', 'pytorch-lightning ',
        'tensorflow',
    ]

    # XXX: this might be a good spot to specify the size of the ResNet
    parameters = {
        'batch_size': [64],
    }

    def __init__(self, batch_size=64):
        # XXX: seed everything correctly
        # https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#reproducibility
        # XXX: modify this with the correct amount of CPUs/GPUs
        self.trainer = Trainer()
        self.batch_size = batch_size

    def set_data(self, dataset):
        self.torch_dataset = dataset
        X = self.torch_dataset.data
        y = self.torch_dataset.targets
        try:
            X = X.numpy()
        except AttributeError:
            pass
        else:
            y = y.numpy()
        self.n_classes = len(np.unique(y))
        self.tf_dataset = tf.data.Dataset.from_tensor_slices((X, y))

    def compute(self, pl_module, tf_model):
        assert xor(pl_module is not None, tf_model is not None), 'You cannot set both pl module and tf model'
        if pl_module is not None:
            loss = self.trainer.test(pl_module)
            loss = loss[0]['train_loss']
        else:
            loss = tf_model.evaluate(self.tf_dataset)
        # XXX: allow to return accuracy as well
        # this will allow to have a more encompassing benchmark that also
        # captures speed on accuracy
        return loss

    def to_dict(self):
        torch_model = models.resnet18(num_classes=self.n_classes)
        data_loader = DataLoader(self.torch_dataset, batch_size=self.batch_size)
        pl_module = BenchPLModule(torch_model, data_loader)
        tf_model = tf.keras.applications.vgg16.VGG16(
            weights=None,
            classes=self.n_classes,
            classifier_activation='softmax',
        )
        tf_dataset = self.tf_dataset.map(
            lambda x, y: (tf.keras.applications.vgg16.preprocess_input(x), y),
        ).batch(self.batch_size).repeat()
        return dict(
            pl_module=pl_module,
            trainer=self.trainer,
            tf_model=tf_model,
            tf_dataset=tf_dataset,
        )
