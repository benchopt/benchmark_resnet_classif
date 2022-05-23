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
        **TFSolver.parameters,
        'lr': [1e-1],
        # parameters are taken from the appendix C.1 from the paper
        # https://arxiv.org/abs/1907.08610
        'coupled_weight_decay': [0.0, 5e-4],
        'momentum': [0.0, 0.9],
        'steps': [[3/10, 6/10, 8/10]],
        'gamma': [0.2],
        'la_steps': [5],
        'la_alpha': [0.8],
        'base_optimizer': ['sgd'],
    }

    base_optimizers_map = {
        'sgd': SGD,
        'adam': Adam,
    }

    def set_objective(self, **kwargs):
        def optimizer_init(lr, weight_decay, momentum, **kwargs):
            base_optimizer_klass = self.base_optimizers_map[
                self.base_optimizer
            ]
            base_optimizer_kwargs = dict(lr=lr)
            if self.base_optimizer == 'sgd':
                base_optimizer_kwargs = dict(
                    **base_optimizer_kwargs,
                    momentum=momentum,
                )
            base_optimizer = base_optimizer_klass(**base_optimizer_kwargs)
            la_optimizer = Lookahead(base_optimizer, **kwargs)
            return la_optimizer

        self.optimizer_klass = optimizer_init
        self.optimizer_kwargs = dict(
            lr=self.lr,
            momentum=self.momentum,
            sync_period=self.la_steps,
            slow_step_size=self.la_alpha,
        )

        super().set_objective(**kwargs)
