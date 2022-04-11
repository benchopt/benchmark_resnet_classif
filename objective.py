from benchopt import BaseObjective, safe_import_context

with safe_import_context() as import_ctx:
    import torch
    import tensorflow as tf
    from pytorch_lightning import Trainer
    from torch.utils.data import DataLoader
    import torchvision.models as models
    BenchPLModule = import_ctx.import_from("torch_helper", "BenchPLModule")
    torch_image_dataset_to_tf_dataset = import_ctx.import_from(
        "tf_helper",
        "torch_image_dataset_to_tf_dataset",
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
        'pytorch', 'torchvision', 'pytorch-lightning ',
        'tensorflow',
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
        accelerator = 'gpu' if torch.cuda.is_available() else None
        self.trainer = Trainer(accelerator=accelerator)
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

    def set_data(self, dataset, test_dataset):
        self.torch_dataset = dataset
        self.torch_test_dataset = test_dataset
        (
            self.tf_dataset,
            self.width,
            self.n_classes,
        ) = torch_image_dataset_to_tf_dataset(self.torch_dataset)
        self.tf_test_dataset, _, _ = torch_image_dataset_to_tf_dataset(
            self.torch_test_dataset,
        )

    def compute(self, model):
        results = dict()
        # XXX: this might be factorized but I think at the cost
        # of readability. Since the only additional framework
        # we might add is Jax atm I think it's ok to have this
        # code not DRY.
        if isinstance(model, tf.keras.models.Model):
            for dataset_name, dataset in zip(
                ["train", "test"], [self.tf_dataset, self.tf_test_dataset]
            ):
                metrics = model.evaluate(
                    self.tf_dataset.batch(self.batch_size),
                    return_dict=True,
                )
                results[dataset_name + "_loss"] = metrics["loss"]
                results[dataset_name + "_acc"] = metrics["accuracy"]
        else:
            for dataset_name, dataset in zip(
                ["train", "test"],
                [self.torch_dataset, self.torch_test_dataset],
            ):
                dataloader = DataLoader(dataset, batch_size=self.batch_size)
                metrics = self.trainer.test(model, dataloaders=dataloader)
                results[dataset_name + "_loss"] = metrics[0]["loss"]
                results[dataset_name + "_acc"] = metrics[0]["acc"]
        results["value"] = results["train_loss"]
        return results

    def get_one_beta(self):
        # XXX: should we have both tf and pl here?
        model = self.get_torch_model()
        return BenchPLModule(model)

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
            torch_dataset=self.torch_dataset,
            tf_model=tf_model,
            tf_dataset=tf_dataset,
        )
