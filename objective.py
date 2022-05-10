import os
import sys

from benchopt import BaseObjective, safe_import_context


with safe_import_context() as import_ctx:
    import joblib
    import torch
    import tensorflow as tf
    from tqdm import tqdm
    import torchvision.models as models
    from torch.utils.data import DataLoader
    from pytorch_lightning import Trainer
    from pytorch_lightning.utilities.seed import seed_everything

    BenchPLModule = import_ctx.import_from("lightning_helper", "BenchPLModule")
    AugmentedDataset = import_ctx.import_from(
        'lightning_helper', 'AugmentedDataset'
    )
    remove_initial_downsample = import_ctx.import_from(
        'torch_resnets', 'remove_initial_downsample'
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

    image_width_cutout = 128

    install_cmd = 'conda'
    requirements = [
        'pip:torch', 'pip:torchvision', 'pip:pytorch-lightning ',
        # TODO: rm below, and fix tests
        'pip:timm', 'pip:tensorflow-datasets', 'pip:tensorflow-addons',
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
        input_width = self.width
        if self.model_type == 'resnet':
            add_kwargs['use_bias'] = False

            # For now 128 is an arbitrary number
            # to differentiate big and small images
            if self.width < self.image_width_cutout:
                input_width = 4*self.width
                add_kwargs['no_initial_downsample'] = True
            else:
                add_kwargs['no_initial_downsample'] = False

        def _model_init_fn():
            model = model_klass(
                weights=None,
                classes=self.n_classes,
                classifier_activation='softmax',
                input_shape=(input_width, input_width, 3),
                **add_kwargs,
            )
            return model
        return _model_init_fn

    def get_torch_model_init_fn(self):
        model_klass = TORCH_MODEL_MAP[self.model_type][self.model_size]

        def _model_init_fn():
            model = model_klass(num_classes=self.n_classes)
            is_resnet = self.model_type == 'resnet'
            is_small_images = self.width < self.image_width_cutout
            if is_resnet and is_small_images:
                model = remove_initial_downsample(model)
            if torch.cuda.is_available():
                model = model.cuda()
            return model
        return _model_init_fn

    def get_lightning_model_init_fn(self):
        torch_model_init_fn = self.get_torch_model_init_fn()

        def _model_init_fn():
            model = torch_model_init_fn()
            return BenchPLModule(model)
        return _model_init_fn

    def get_model_init_fn(self, framework):
        if framework == 'tensorflow':
            return self.get_tf_model_init_fn()
        elif framework == 'lightning':
            return self.get_lightning_model_init_fn()
        elif framework == 'pytorch':
            return self.get_torch_model_init_fn()
        else:
            raise ValueError(f"No framework named {framework}")

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
            accelerator="auto", strategy="noteardown", max_epochs=-1,
            enable_checkpointing=False,
            enable_model_summary=False
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
            elif self.framework in ['lightning', 'pytorch']:
                # Don't use multiple workers on OSX as this leads to deadlock
                # in the CI.
                # XXX - try to come up with better way to set this.
                system = os.environ.get('RUNNER_OS', sys.platform)
                is_mac = system in ['darwin', 'macOS']
                num_workers = min(10, joblib.cpu_count()) if not is_mac else 0
                persistent_workers = num_workers > 0

                if dataset_name == 'train':
                    data = AugmentedDataset(data, None, self.normalization)

                self._datasets[dataset_name] = DataLoader(
                    data, batch_size=test_batch_size,
                    num_workers=num_workers,
                    persistent_workers=persistent_workers,
                    pin_memory=True
                )

    def compute(self, model):
        results = dict()
        for dataset_name, dataset in self._datasets.items():

            if self.framework == 'tensorflow':
                metrics = model.evaluate(dataset, return_dict=True)
            elif self.framework == 'lightning':
                metrics = self.trainer.test(model, dataloaders=dataset)[0]
            elif self.framework == 'pytorch':
                metrics = self.eval_torch(model, dataloader=dataset)

            results[dataset_name + "_loss"] = metrics["loss"]
            acc_name = "accuracy" if self.framework == 'tensorflow' else "acc"
            results[dataset_name + "_err"] = 1 - metrics[acc_name]

        results["value"] = results["train_loss"]
        return results

    def eval_torch(self, model, dataloader):

        model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        res = {'loss': 0., 'acc': 0, 'n_samples': 0}
        with torch.no_grad():
            for X, y in tqdm(dataloader):
                if torch.cuda.is_available():
                    X, y = X.cuda(), y.cuda()
                res['n_samples'] += len(X)
                y_proba = model(X)
                res['loss'] += criterion(y_proba, y).item()
                res['acc'] += (y_proba.argmax(axis=1) == y).sum().item()
        res['loss'] /= res['n_samples']
        res['acc'] /= res['n_samples']

        model.train()
        return res

    def to_dict(self):
        return dict(
            model_init_fn=self.get_one_beta,
            dataset=self.dataset,
            normalization=self.normalization,
            framework=self.framework,
        )
