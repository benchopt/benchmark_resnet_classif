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
        'batch_size': [64],
        'data_aug': [False, True],
        'rand_aug': [False, True],
    }

    def skip(self, model, dataset):
        if not isinstance(model, BenchPLModule):
            return True, 'Not a PT dataset'
        if self.rand_aug and not self.data_aug:
            return True, 'Data augmentation not activated for RA'
        return False, None

    def __init__(self, **parameters):
        aug_list = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        if self.rand_aug:
            # we put magnitude to 10, to copy TF models
            aug_list.append(transforms.RandomAugment(magnitude=10))
        self.data_aug_transform = transforms.Compose(aug_list)

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
