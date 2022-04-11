from benchopt import BaseObjective, safe_import_context

with safe_import_context() as import_ctx:
    from pytorch_lightning import Trainer
    import tensorflow as tf
    from torch.utils.data import DataLoader
    import torchvision.models as models
    BenchPLModule = import_ctx.import_from("torch_helper", "BenchPLModule")
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

    name = "ConvNet classification fitting"
    is_convex = False

    install_cmd = 'conda'
    requirements = [
        'pytorch', 'torchvision', 'pytorch-lightning ',
        'tensorflow', 'tensorflow-datasets',
    ]

    # XXX: this might be a good spot to specify the size of the ResNet
    parameters = {
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

    def skip(self):
        if self.framework == 'tensorflow' and self.image_width < 32:
            return True, 'images too small for TF networks'
        return False, None

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

    def set_data(
        self,
        dataset,
        test_dataset,
        n_samples_train,
        n_samples_test,
        image_width,
        n_classes,
        framework,
    ):
        self.dataset = dataset
        self.test_dataset = test_dataset
        self.n_samples_train = n_samples_train
        self.n_samples_test = n_samples_test
        self.width = image_width
        self.n_classes = n_classes
        self.framework = framework

    def compute(self, model):
        results = dict()
        # XXX: this might be factorized but I think at the cost
        # of readability. Since the only additional framework
        # we might add is Jax atm I think it's ok to have this
        # code not DRY.
        if self.framework == 'tensorflow':
            for dataset_name, dataset in zip(
                ["train", "test"], [self.dataset, self.test_dataset]
            ):
                metrics = model.evaluate(
                    # TODO: optimize this with prefetching
                    dataset.batch(self.batch_size),
                    return_dict=True,
                )
                results[dataset_name + "_loss"] = metrics["loss"]
                results[dataset_name + "_acc"] = metrics["accuracy"]
        elif self.framework == 'pytorch':
            for dataset_name, dataset in zip(
                ["train", "test"],
                [self.dataset, self.test_dataset],
            ):
                dataloader = DataLoader(dataset, batch_size=self.batch_size)
                metrics = self.trainer.test(model, dataloaders=dataloader)
                results[dataset_name + "_loss"] = metrics[0]["loss"]
                results[dataset_name + "_acc"] = metrics[0]["acc"]
        results["value"] = results["train_loss"]
        return results

    def get_one_beta(self):
        # XXX: should we have both tf and pl here?
        if self.framework == 'tensorflow':
            model = self.get_tf_model()
        elif self.framework == 'pytorch':
            model = self.get_torch_model()
            model = BenchPLModule(model)
        return model

    def to_dict(self):
        # XXX: make sure to skip the small datasets when using vgg
        model = self.get_one_beta()
        return dict(
            model=model,
            dataset=self.dataset,
        )
