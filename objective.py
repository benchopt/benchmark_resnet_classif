from benchopt import BaseObjective, safe_import_context

with safe_import_context() as import_ctx:
    from pytorch_lightning import Trainer
    import tensorflow as tf
    from torch.utils.data import DataLoader
    import torchvision.models as models
    BenchPLModule = import_ctx.import_from("torch_helper", "BenchPLModule")
    torch_image_dataset_to_tf_dataset = import_ctx.import_from(
        "tf_helper",
        "torch_image_dataset_to_tf_dataset",
    )


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
        "batch_size": [64],
    }

    def __init__(self, batch_size=64):
        # XXX: seed everything correctly
        # https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#reproducibility
        # XXX: modify this with the correct amount of CPUs/GPUs
        self.trainer = Trainer()
        self.batch_size = batch_size

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
        results["value"] = results["train_acc"]
        return results

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
        # XXX: for the tf model we might consider other options like
        # https://github.com/keras-team/keras-contrib
        # But it looks dead, and not moved to tf-addons
        # Same for https://github.com/qubvel/classification_models
        if self.width < 32:
            # Because vgg16 doesn't support small images
            # we might need to handle this some other way
            # when we specify the model size and type in the objective param
            # by skipping the MNIST dataset for the vgg models
            tf_model = tf_dataset = None
        else:
            tf_model = tf.keras.applications.vgg16.VGG16(
                weights=None,
                classes=self.n_classes,
                classifier_activation='softmax',
                input_shape=(self.width, self.width, 3),
            )
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
