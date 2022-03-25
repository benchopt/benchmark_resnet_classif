from benchopt import BaseObjective, safe_import_context

with safe_import_context() as import_ctx:
    from pytorch_lightning import Trainer

    from torch.utils.data import DataLoader
    import torchvision.models as models
    BenchPLModule = import_ctx.import_from('torch_helper', 'BenchPLModule')


class Objective(BaseObjective):
    """Classification objective"""
    name = "ResNet classification fitting"
    is_convex = False

    install_cmd = 'conda'
    requirements = [
        'pytorch', 'torchvision', 'pytorch-lightning '
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
        self.dataset = dataset

    def compute(self, pl_module):
        loss = self.trainer.test(pl_module)
        # XXX: allow to return accuracy as well
        # this will allow to have a more encompassing benchmark that also
        # captures speed on accuracy
        return loss[0]['train_loss']

    def to_dict(self):
        model = models.resnet18()
        data_loader = DataLoader(self.dataset, batch_size=self.batch_size)
        pl_module = BenchPLModule(model, data_loader)
        return dict(
            pl_module=pl_module,
            trainer=self.trainer,
        )
