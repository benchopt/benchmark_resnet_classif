from benchopt import BaseObjective, safe_import_context


with safe_import_context() as import_ctx:
    import numpy as np
    from pytorch_lightning import Trainer
    import tensorflow as tf
    from torch.utils.data import DataLoader
    import torchvision.models as models
    BenchPLModule = import_ctx.import_from('torch_helper', 'BenchPLModule')
    TFResNet18 = import_ctx.import_from('tf_resnets', 'ResNet18')
    TFResNet34 = import_ctx.import_from('tf_resnets', 'ResNet34')
    TFResNet50 = import_ctx.import_from('tf_resnets', 'ResNet50')


TF_MODEL_MAP = {
    'resnet': {
        '18': TFResNet18,
        '34': TFResNet34,
        '50': TFResNet50,
    },
    'vgg': {
        '16': tf.keras.applications.vgg16.VGG16,
    }
}

TORCH_MODEL_MAP = {
    'resnet': {
        '18': models.resnet18,
        '34': models.resnet34,
        '50': models.resnet50,
    },
    'vgg': {
        '16': models.vgg16,
    }
}

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
        'model_type, model_size': [
            ('resnet', '18'),
            ('resnet', '34'),
            ('resnet', '50'),
            ('vgg', '16'),
        ]
    }

    def __init__(self, batch_size=64, model_type='resnet', model_size='18'):
        # XXX: seed everything correctly
        # https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#reproducibility
        # XXX: modify this with the correct amount of CPUs/GPUs
        self.trainer = Trainer()
        self.batch_size = batch_size
        self.model_type = model_type
        self.model_size = model_size

    def get_tf_model(self):
        model_klass = TF_MODEL_MAP[self.model_type][self.model_size]
        add_kwargs = {}
        if self.model_type == 'resnet':
            add_kwargs['use_bias'] = False
        model = model_klass(
            weights=None,
            classes=self.n_classes,
            classifier_activation='softmax',
            input_shape=(self.width, self.width, 3),
            **add_kwargs,
        )
        return model

    def get_torch_model(self):
        model_klass = TORCH_MODEL_MAP[self.model_type][self.model_size]
        model = model_klass(num_classes=self.n_classes)
        return model

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
            try:
                y = self.torch_dataset.targets
            except AttributeError:
                y = self.torch_dataset.labels
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
        model = self.get_torch_model()
        data_loader = DataLoader(
            self.torch_dataset,
            batch_size=self.batch_size,
        )
        return BenchPLModule(model, data_loader)

    def to_dict(self):
        # XXX: make sure to skip the small datasets when using vgg
        pl_module = self.get_one_beta()
        tf_model = self.get_tf_model()
        tf_dataset = self.tf_dataset.batch(
            self.batch_size,
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
