import os
import sys

from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion


with safe_import_context() as import_ctx:

    import joblib
    from timm.data.mixup import Mixup
    import torch
    from torch.utils.data._utils.collate import default_collate
    from torchvision import transforms
    from tqdm import tqdm

    AugmentedDataset = import_ctx.import_from(
        'lightning_helper', 'AugmentedDataset'
    )


class TorchSolver(BaseSolver):
    """Pytorch base solver"""

    stopping_criterion = SufficientProgressCriterion(
        patience=20, strategy='callback'
    )

    parameters = {
        'batch_size': [128],
        'data_aug': [False, True],
        'rand_aug': [False, True],
        'mix': [False, True],
        'lr_schedule': [None, 'step', 'cosine'],
    }

    install_cmd = 'conda'
    requirements = ['timm']

    def skip(self, model_init_fn, dataset, normalization, framework):
        if framework != 'pytorch':
            return True, 'Not a torch dataset/objective'
        if self.rand_aug and not self.data_aug:
            return True, 'Data augmentation not activated for RA'
        coupled_wd = getattr(self, 'coupled_weight_decay', 0.0)
        decoupled_wd = getattr(self, 'decoupled_weight_decay', 0.0)
        if coupled_wd and decoupled_wd:
            return True, 'Cannot use both decoupled and coupled weight decay'
        return False, None

    def set_objective(self, model_init_fn, dataset, normalization, framework):
        self.dataset = dataset
        self.model_init_fn = model_init_fn
        self.normalization = normalization
        self.framework = framework

        if self.mix:
            self.mixup_fn = lambda batch: Mixup(
                mixup_alpha=0.1,
                cutmix_alpha=1.0,
                num_classes=dataset.n_classes,
            )(*default_collate(batch))
            # TODO: change below
            self.model.loss_type = 'bce'
        if self.data_aug:
            aug_list = [
                # TODO: we need to change the size
                # to fit the dataset
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
            if self.rand_aug:
                # we put magnitude to 10, to copy TF models
                aug_list.insert(0, transforms.RandAugment(magnitude=10))
            data_aug_transform = transforms.Compose(aug_list)
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
            pin_memory=True, shuffle=True,
            collate_fn=self.mixup_fn if self.mix else None,
        )

    def set_lr_schedule_and_optimizer(self, model, max_epochs=200):
        optimizer = self.optimizer_klass(
            model.parameters(),
            **self.optimizer_kwargs,
        )
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
        else:
            class NoOpScheduler:
                def step(self):
                    ...

            scheduler = NoOpScheduler()
        return optimizer, scheduler

    @staticmethod
    def get_next(stop_val):
        return stop_val + 1

    def run(self, callback):
        # model weight initialization
        model = self.model_init_fn()
        criterion = torch.nn.CrossEntropyLoss()

        # optimizer and lr schedule init
        max_epochs = callback.stopping_criterion.max_runs
        optimizer, lr_schedule = self.set_lr_schedule_and_optimizer(
            model,
            max_epochs,
        )
        # Initial evaluation
        while callback(model):
            for X, y in tqdm(self.dataloader):
                if torch.cuda.is_available():
                    X, y = X.cuda(), y.cuda()
                optimizer.zero_grad()
                loss = criterion(model(X), y)
                loss.backward()

                optimizer.step()
            lr_schedule.step()

        self.model = model

    def get_result(self):
        return self.model
