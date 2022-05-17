from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    from tensorflow_addons.optimizers import Lookahead
    from tensorflow.keras.optimizers import SGD, Adam

TFSolver = import_ctx.import_from('tf_solver', 'TFSolver')


class Solver(TFSolver):
    """Lookahead solver"""
    name = 'Lookahead-tf'

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'lr': [1e-1],
        # parameters are taken from the appendix C.1 from the paper
        # https://arxiv.org/abs/1907.08610
        'la_steps': [5],
        'la_alpha': [0.8],
        'base_optimizer': ['sgd'],
        **TFSolver.parameters,
    }

    base_optimizers_map = {
        'sgd': SGD,
        'adam': Adam,
    }

    def set_objective(self, **kwargs):
        def optimizer_init(lr, weight_decay, **kwargs):
            base_optimizer_klass = self.base_optimizers_map[
                self.base_optimizer
            ]
            base_optimizer = base_optimizer_klass(lr=lr)
            la_optimizer = Lookahead(base_optimizer, **kwargs)
            return la_optimizer

        self.optimizer_klass = optimizer_init
        self.optimizer_kwargs = dict(
            lr=self.lr,
            sync_period=self.la_steps,
            slow_step_size=self.la_alpha,
        )

        super().set_objective(**kwargs)
