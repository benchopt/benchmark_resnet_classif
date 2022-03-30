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
    is_convex = False

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
        try:
            X = self.torch_dataset.data
        except AttributeError:
            _loader = DataLoader(
                self.torch_dataset,
                batch_size=len(self.torch_dataset),
            )
            _sample = next(iter(_loader))
            X = _sample[0]
            y = _sample[1]
        else:
            y = self.torch_dataset.targets
        try:
            X = X.numpy()
        except AttributeError:
            pass
        else:
            y = y.numpy()
        if X.shape[1] in [1, 3]:
            # reshape X from NCHW to NHWC
            X = np.transpose(X, (0, 2, 3, 1))
        self.width = X.shape[1]
        self.n_classes = len(np.unique(y))
        if not isinstance(y[0], np.ndarray) or not len(y) > 1:
            y = tf.one_hot(y, self.n_classes)
        self.tf_dataset = tf.data.Dataset.from_tensor_slices((X, y))

    def compute(self, model):
        if isinstance(model, tf.keras.models.Model):
            loss = model.evaluate(self.tf_dataset.batch(self.batch_size))
        else:
            loss = self.trainer.test(model)
            loss = loss[0]['train_loss']
        # XXX: allow to return accuracy as well
        # this will allow to have a more encompassing benchmark that also
        # captures speed on accuracy
        return loss

    def get_one_beta(self):
        # XXX: should we have both tf and pl here?
        model = models.resnet18(num_classes=self.n_classes)
        data_loader = DataLoader(
            self.torch_dataset,
            batch_size=self.batch_size,
        )
        return BenchPLModule(model, data_loader)

    def to_dict(self):
        pl_module = self.get_one_beta()
        tf_model = tf.keras.applications.vgg16.VGG16(
            weights=None,
            classes=self.n_classes,
            classifier_activation='softmax',
            input_shape=(self.width, self.width, 3),
        )
        tf_dataset = self.tf_dataset.batch(
            self.batch_size,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        ).map(
            lambda x, y: (tf.keras.applications.vgg16.preprocess_input(x), y),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        ).prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE,
        )
        return dict(
            pl_module=pl_module,
            trainer=self.trainer,
            tf_model=tf_model,
            tf_dataset=tf_dataset,
        )
