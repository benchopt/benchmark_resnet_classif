import os
import sys
from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion


with safe_import_context() as import_ctx:

    import joblib
    import torch
    from torchvision import transforms
    from pytorch_lightning import Trainer

    BenchoptCallback = import_ctx.import_from(
        'lightning_helper', 'BenchoptCallback'
    )
    AugmentedDataset = import_ctx.import_from(
        'lightning_helper', 'AugmentedDataset'
    )


class LightningSolver(BaseSolver):
    """Pytorch Lightning base solver"""

    stopping_criterion = SufficientProgressCriterion(
        patience=20, strategy='callback'
    )

    parameters = {
        'batch_size': [128],
        'data_aug': [False, True],
        'lr_schedule': [None, 'step', 'cosine'],
    }

    def skip(
        self,
        model_init_fn,
        dataset,
        normalization,
        framework,
        symmetry,
        image_width,
    ):
        if framework != 'lightning':
            return True, 'Not a PL dataset/objective'
        coupled_wd = getattr(self, 'coupled_weight_decay', 0.0)
        decoupled_wd = getattr(self, 'decoupled_weight_decay', 0.0)
        if coupled_wd and decoupled_wd:
            return True, 'Cannot use both decoupled and coupled weight decay'
        return False, None

    def set_objective(
        self,
        model_init_fn,
        dataset,
        normalization,
        framework,
        symmetry,
        image_width,
    ):
        self.dataset = dataset
        self.model_init_fn = model_init_fn
        self.normalization = normalization
        self.framework = framework
        self.symmetry = symmetry
        self.image_width = image_width

        if self.data_aug:
            data_aug_list = [
                transforms.RandomCrop(self.image_width, padding=4),
            ]
            if self.symmetry is not None and 'horizontal' in self.symmetry:
                data_aug_list.append(transforms.RandomHorizontalFlip())
            data_aug_transform = transforms.Compose(data_aug_list)
        else:
            data_aug_transform = None
        self.dataset = AugmentedDataset(
            self.dataset,
            data_aug_transform,
            self.normalization,
        )

        # TODO: num_worker should not be hard coded. Finding a sensible way to
        # set this value is necessary here.
        system = os.environ.get('RUNNER_OS', sys.platform)
        is_mac = system in ['darwin', 'macOS']
        num_workers = min(10, joblib.cpu_count()) if not is_mac else 0
        persistent_workers = num_workers > 0
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.batch_size,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            pin_memory=True, shuffle=True
        )

    def set_lr_schedule_and_optimizer(self, max_epochs=200):
        optimizer = self.optimizer_klass(
            self.model.parameters(),
            **self.optimizer_kwargs,
        )
        if self.lr_schedule is None:
            self.model.configure_optimizers = lambda: optimizer
            return
        if self.lr_schedule == 'step':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[max_epochs//2, max_epochs*3//4],
                gamma=0.1,
            )
        elif self.lr_schedule == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max_epochs,
            )
        self.model.configure_optimizers = lambda: (
            [optimizer],
            [scheduler],
        )

    @staticmethod
    def get_next(stop_val):
        return stop_val + 1

    def run(self, callback):
        # model weight initialization
        self.model = self.model_init_fn()
        # optimizer and lr schedule init
        max_epochs = callback.stopping_criterion.max_runs
        self.set_lr_schedule_and_optimizer(max_epochs)
        # Initial evaluation
        callback(self.model)

        # Setup the trainer
        # TODO: for now, we are limited to 1 device due to pytorch_lightning
        # bad interaction with benchopt. Removing this limitation would be
        # nice to allow multi-GPU training.
        trainer = Trainer(
            max_epochs=-1, callbacks=[BenchoptCallback(callback)],
            accelerator="auto", devices=1,
            enable_checkpointing=False,
            enable_model_summary=False,
        )
        trainer.fit(self.model, train_dataloaders=self.dataloader)

    def get_result(self):
        return self.model
