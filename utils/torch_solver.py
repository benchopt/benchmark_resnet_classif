import os
import platform
from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion


with safe_import_context() as import_ctx:

    import joblib
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

    stopping_criterion = SufficientProgressCriterion(
        patience=20, strategy='callback'
    )

    parameters = {
        'batch_size': [64],
        'data_aug': [False, True],
    }

    def skip(self, model_init_fn, dataset):
        if not isinstance(dataset, torch.utils.data.Dataset):
            return True, 'Not a PT dataset'
        return False, None

    def __init__(self, **parameters):
        self.data_aug_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])

    def set_objective(self, model_init_fn, dataset):
        self.model_init_fn = model_init_fn
        self.dataset = dataset  # we use this in order
        # to access some elements from the trainer when
        # initializing it below
        if self.data_aug:
            self.dataset = AugmentedDataset(
                self.dataset,
                self.data_aug_transform,
            )
        # TODO: num_worker should not be hard coded. Finding a sensible way to
        # set this value is necessary here.
        system = os.environ.get('RUNNER_OS', platform.system())
        is_mac = system == 'Darwin' or system == 'macOS'
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.batch_size,
            num_workers=min(10, joblib.cpu_count()) if not is_mac else 0,
        )

    @staticmethod
    def get_next(stop_val):
        return stop_val + 1

    def run(self, callback):
        # model weight initialization
        self.model = BenchPLModule(self.model_init_fn())
        # optimizer init
        self.model.configure_optimizers = lambda: self.optimizer_klass(
            self.model.parameters(),
            **self.optimizer_kwargs,
        )
        # Initial evaluation
        callback(self.model)

        # Setup the trainer
        # TODO: for now, we are limited to 1 device due to pytorch_lightning
        # bad interaction with benchopt. Removing this limitation would be
        # nice to allow multi-GPU training.
        trainer = Trainer(
            max_epochs=-1, callbacks=[BenchoptCallback(callback)],
            accelerator="auto", devices=1
        )
        trainer.fit(self.model, train_dataloaders=self.dataloader)

    def get_result(self):
        return self.model
