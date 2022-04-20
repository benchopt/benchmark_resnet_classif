from benchopt import BaseSolver, safe_import_context


with safe_import_context() as import_ctx:

    import torch
    from torchvision import transforms
    from pytorch_lightning import Trainer
    BenchoptCallback = import_ctx.import_from(
        'torch_helper', 'BenchoptCallback'
    )
    BenchPLModule = import_ctx.import_from(
        'torch_helper', 'BenchPLModule'
    )
    AugmentedDataset = import_ctx.import_from(
        'torch_helper', 'AugmentedDataset'
    )


class TorchSolver(BaseSolver):
    """Torch base solver"""

    stopping_strategy = 'callback'

    parameters = {
        'lr': [1e-3],
        'batch_size': [64],
        'data_aug': [False, True],
        'lr_schedule': [None, 'step', 'cosine'],
        'weight_decay': [0.0, 1e-4, 0.02],
    }

    def skip(self, model, dataset):
        if not isinstance(model, BenchPLModule):
            return True, 'Not a PT dataset'
        return False, None

    def __init__(self, **parameters):
        self.data_aug_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])

    def set_objective(self, model, dataset):
        self.model = model
        self.dataset = dataset  # we use this in order
        # to access some elements from the trainer when
        # initializing it below
        if self.data_aug:
            self.dataset = AugmentedDataset(
                self.dataset,
                self.data_aug_transform,
            )
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.batch_size
        )

    def set_lr_schedule_and_optimizer(self, optimizer):
        if self.lr_schedule is None:
            self.model.configure_optimizers = lambda: optimizer
            return
        if self.lr_schedule == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=30,
                gamma=0.1,
            )
        elif self.lr_schedule == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=200,
            )
        self.model.configure_optimizers = lambda: (
            optimizer,
            scheduler,
        )

    @staticmethod
    def get_next(stop_val):
        return stop_val + 1

    def run(self, callback):
        # Initial evaluation
        callback(self.model)

        # Setup the trainer
        trainer = Trainer(
            max_epochs=-1, callbacks=[BenchoptCallback(callback)]
        )
        trainer.fit(self.model, train_dataloaders=self.dataloader)

    def get_result(self):
        return self.model
