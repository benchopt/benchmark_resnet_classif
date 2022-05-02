import os
import sys

from benchopt import BaseObjective, safe_import_context


with safe_import_context() as import_ctx:
    import joblib
    import tensorflow as tf

    import torchvision.models as models
    from torch.utils.data import DataLoader
    from pytorch_lightning import Trainer
    from pytorch_lightning.utilities.seed import seed_everything

    BenchPLModule = import_ctx.import_from("torch_helper", "BenchPLModule")
    AugmentedDataset = import_ctx.import_from(
        'torch_helper', 'AugmentedDataset'
    )
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
        'pip:tensorflow',
    ]

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
        normalization,
    ):
        if framework == 'tensorflow' and image_width < 32:
            return True, 'images too small for TF networks'
        return False, None

    def get_tf_model_init_fn(self):
        model_klass = TF_MODEL_MAP[self.model_type][self.model_size]
        add_kwargs = {}
        if self.model_type == 'resnet':
            add_kwargs['use_bias'] = False

        # For now 128 is an arbitrary number
        # to differentiate big and small images
        if self.width < 128:
            input_width = 4*self.width
            no_initial_downsample = True
        else:
            input_width = self.width
            no_initial_downsample = False

        def _model_init_fn():
            model = model_klass(
                weights=None,
                classes=self.n_classes,
                classifier_activation='softmax',
                input_shape=(input_width, input_width, 3),
                no_initial_downsample=no_initial_downsample,
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
        normalization,
    ):
        self.dataset = dataset
        self.test_dataset = test_dataset
        self.n_samples_train = n_samples_train
        self.n_samples_test = n_samples_test
        self.width = image_width
        self.n_classes = n_classes
        self.framework = framework
        self.normalization = normalization

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
        test_batch_size = 100
        self._datasets = {}
        for dataset_name, data in [('train', self.dataset),
                                   ('test', self.test_dataset)]:
            if self.framework == 'tensorflow':
                ds = data.batch(test_batch_size)
                if dataset_name == 'train':
                    ds = ds.map(
                        lambda x, y: (self.normalization(x), y),
                        num_parallel_calls=tf.data.experimental.AUTOTUNE,
                    )
                self._datasets[dataset_name] = ds
            elif self.framework == 'pytorch':
                # Don't use multiple workers on OSX as this leads to deadlock
                # in the CI.
                # XXX - try to come up with better way to set this.
                system = os.environ.get('RUNNER_OS', sys.platform)
                is_mac = system in ['darwin', 'macOS']
                num_workers = min(10, joblib.cpu_count()) if not is_mac else 0

                if dataset_name == 'train':
                    data = AugmentedDataset(data, None, self.normalization)

                self._datasets[dataset_name] = DataLoader(
                    data, batch_size=test_batch_size,
                    num_workers=num_workers, persistent_workers=True,
                    pin_memory=True
                )

    def compute(self, model):
        results = dict()
        for dataset_name, dataset in self._datasets.items():

            if self.framework == 'tensorflow':
                metrics = model.evaluate(dataset, return_dict=True)
            elif self.framework == 'pytorch':
                metrics = self.trainer.test(model, dataloaders=dataset)[0]

            results[dataset_name + "_loss"] = metrics["loss"]
            acc_name = "accuracy" if self.framework == 'tensorflow' else "acc"
            results[dataset_name + "_err"] = 1 - metrics[acc_name]

        results["value"] = results["train_loss"]
        return results

    def to_dict(self):
        return dict(
            model_init_fn=self.get_one_beta,
            dataset=self.dataset,
            normalization=self.normalization,
        )
