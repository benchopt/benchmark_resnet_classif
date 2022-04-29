from benchopt import BaseObjective, safe_import_context

with safe_import_context() as import_ctx:
    from pytorch_lightning.utilities.seed import seed_everything
    import tensorflow as tf
    from pytorch_lightning import Trainer
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
        'pip:pytorch', 'pip:torchvision', 'pip:pytorch-lightning ',
        'pip:tensorflow', 'pip:tensorflow-datasets',
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

    def skip(
        self,
        dataset,
        test_dataset,
        n_samples_train,
        n_samples_test,
        image_width,
        n_classes,
        framework,
    ):
        if framework == 'tensorflow' and image_width < 32:
            return True, 'images too small for TF networks'
        return False, None

    def get_tf_model_init_fn(self):
        model_klass = TF_MODEL_MAP[self.model_type][self.model_size]
        add_kwargs = {}
        if self.model_type == 'resnet':
            add_kwargs['use_bias'] = False

        def _model_init_fn():
            model = model_klass(
                weights=None,
                classes=self.n_classes,
                classifier_activation='softmax',
                input_shape=(self.width, self.width, 3),
                **add_kwargs,
            )
            return model
        return _model_init_fn

    def get_torch_model_init_fn(self):
        model_klass = TORCH_MODEL_MAP[self.model_type][self.model_size]

        def _model_init_fn():
            model = model_klass(num_classes=self.n_classes)
            return BenchPLModule(model)
        return _model_init_fn

    def get_model_init_fn(self, framework):
        if framework == 'tensorflow':
            return self.get_tf_model_init_fn()
        elif framework == 'pytorch':
            return self.get_torch_model_init_fn()

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

        # Get the model initializer
        self.get_one_beta = self.get_model_init_fn(framework)

        # seeding for the models
        # XXX: This should be changed once benchopt/benchopt#342 is merged
        tf.random.set_seed(0)
        seed_everything(0, workers=True)

        # https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#reproducibility
        # XXX: modify this with the correct amount of CPUs/GPUs
        self.trainer = Trainer(
            accelerator="auto", strategy="noteardown", max_epochs=-1
        )

        # Set the batch size for the test dataloader
        self._test_batch_size = 100

    def compute(self, model):
        results = dict()
        for dataset_name, dataset in zip(
            ["train", "test"], [self.dataset, self.test_dataset]
        ):
            if self.framework == 'tensorflow':
                metrics = model.evaluate(
                    dataset.batch(self._test_batch_size),
                    return_dict=True,
                )
            elif self.framework == 'pytorch':
                dataloader = DataLoader(
                    dataset, batch_size=self._test_batch_size
                )
                metrics = self.trainer.test(model, dataloaders=dataloader)[0]
            results[dataset_name + "_loss"] = metrics["loss"]
            acc_name = "accuracy" if self.framework == 'tensorflow' else "acc"
            results[dataset_name + "_acc"] = metrics[acc_name]

        results["value"] = results["train_loss"]
        return results

    def to_dict(self):
        # XXX: make sure to skip the small datasets when using vgg
        return dict(
            model_init_fn=self.get_one_beta,
            dataset=self.dataset,
        )
