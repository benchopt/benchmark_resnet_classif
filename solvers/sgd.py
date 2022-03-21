from benchopt import BaseSolver
from pytorch_lightning import Trainer
from torch.optim import SGD


class Solver(BaseSolver):
    """Stochastic Gradient descent solver"""
    name = 'SGD'

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'nesterov': [False, True],
        'lr': [1e-3],
        'momentum': [0, 0.9],
    }

    def skip(self, pl_module, trainer):
        if not self.momentum and self.nesterov:
            return True, 'Nesterov cannot be used without momentum'

    def set_objective(self, pl_module, trainer):
        self.pl_module = pl_module
        self.main_trainer = trainer  # we use this in order
        # to access some elements from the trainer when3
        # initializing it below
        self.pl_module.configure_optimizers = lambda: SGD(
            self.pl_module.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            nesterov=self.nesterov,
        )

    def run(self, n_iter):
        trainer = Trainer(max_steps=n_iter)
        trainer.fit(self.pl_module)

    def get_result(self):
        return self.pl_module
