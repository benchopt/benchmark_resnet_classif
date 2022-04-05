from benchopt import BaseObjective, safe_import_context

with safe_import_context() as import_ctx:
    from pytorch_lightning import Trainer

    from torch.utils.data import DataLoader
    import torchvision.models as models
    BenchPLModule = import_ctx.import_from("torch_helper", "BenchPLModule")


class Objective(BaseObjective):
    """Classification objective"""

    name = "ResNet classification fitting"
    is_convex = False

    install_cmd = "conda"
    requirements = ["pytorch", "torchvision", "pytorch-lightning "]

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
        self.dataset = dataset
        self.test_dataset = test_dataset

    def compute(self, pl_module):
        results = dict()
        for dataset_name, dataset in zip(
            ["train", "test"], [self.dataset, self.test_dataset]
        ):
            dataloader = DataLoader(dataset, batch_size=self.batch_size)
            metrics = self.trainer.test(pl_module, dataloaders=dataloader)
            results[dataset_name + "_loss"] = metrics[0]["loss"]
            results[dataset_name + "_acc"] = metrics[0]["acc"]
        results["value"] = results["train_acc"]
        return results

    def get_one_beta(self):
        model = models.resnet18()
        data_loader = DataLoader(self.dataset, batch_size=self.batch_size)
        return BenchPLModule(model, data_loader)

    def to_dict(self):
        pl_module = self.get_one_beta()
        return dict(
            pl_module=pl_module,
            trainer=self.trainer,
        )
