from benchopt import BaseSolver
from pytorch_lightning import Trainer


class BaseNNSolver(BaseSolver):
    """NN solver"""
    name = 'NN solver'

    def set_objective(self, pl_module, trainer):
        self.pl_module = pl_module
        self.main_trainer = trainer  # we use this in order
        # to access some elements from the trainer when3
        # initializing it below


    def run(self, n_iter):
        trainer = Trainer(max_steps=n_iter)
        trainer.fit(self.pl_module)

    def get_result(self):
        return self.pl_module
