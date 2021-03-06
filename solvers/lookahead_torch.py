from collections import defaultdict

from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    # this is taken from the official implementation
    # https://github.com/michaelrzhang/lookahead/blob/master/lookahead_pytorch.py
    # only changes are for flake8
    import torch
    from torch.optim import SGD, Adam
    from torch.optim.optimizer import Optimizer
    from tqdm import tqdm

    class Lookahead(Optimizer):
        r"""PyTorch implementation of the lookahead wrapper.
        Lookahead Optimizer: https://arxiv.org/abs/1907.08610
        """

        def __init__(
            self,
            optimizer,
            la_steps=5,
            la_alpha=0.8,
            pullback_momentum="none",
        ):
            """optimizer: inner optimizer
            la_steps (int): number of lookahead steps
            la_alpha (float): linear interpolation factor. 1.0 recovers
                the inner optimizer.
            pullback_momentum (str): change to inner optimizer momentum on
                interpolation update
            """
            self.optimizer = optimizer
            self._la_step = 0  # counter for inner optimizer
            self.la_alpha = la_alpha
            self._total_la_steps = la_steps
            pullback_momentum = pullback_momentum.lower()
            assert pullback_momentum in ["reset", "pullback", "none"]
            self.pullback_momentum = pullback_momentum

            self.state = defaultdict(dict)

            # Cache the current optimizer parameters
            for group in optimizer.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    param_state['cached_params'] = torch.zeros_like(p.data)
                    param_state['cached_params'].copy_(p.data)
                    if self.pullback_momentum == "pullback":
                        param_state['cached_mom'] = torch.zeros_like(p.data)

        def __getstate__(self):
            return {
                'state': self.state,
                'optimizer': self.optimizer,
                'la_alpha': self.la_alpha,
                '_la_step': self._la_step,
                '_total_la_steps': self._total_la_steps,
                'pullback_momentum': self.pullback_momentum
            }

        def zero_grad(self):
            self.optimizer.zero_grad()

        def get_la_step(self):
            return self._la_step

        def state_dict(self):
            return self.optimizer.state_dict()

        def load_state_dict(self, state_dict):
            self.optimizer.load_state_dict(state_dict)

        def _backup_and_load_cache(self):
            """Useful for performing evaluation on the slow weights
                (which typically generalize better)
            """
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    param_state['backup_params'] = torch.zeros_like(p.data)
                    param_state['backup_params'].copy_(p.data)
                    p.data.copy_(param_state['cached_params'])

        def _clear_and_load_backup(self):
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    p.data.copy_(param_state['backup_params'])
                    del param_state['backup_params']

        @property
        def param_groups(self):
            return self.optimizer.param_groups

        def step(self, closure=None):
            """Performs a single Lookahead optimization step.
            Arguments:
                closure (callable, optional): A closure that reevaluates
                    the model and returns the loss.
            """
            loss = self.optimizer.step(closure)
            self._la_step += 1

            if self._la_step >= self._total_la_steps:
                self._la_step = 0
                # Lookahead and cache the current optimizer parameters
                for group in self.optimizer.param_groups:
                    for p in group['params']:
                        param_state = self.state[p]
                        p.data.mul_(self.la_alpha).add_(
                            param_state['cached_params'],
                            alpha=1.0 - self.la_alpha,
                        )  # crucial line
                        param_state['cached_params'].copy_(p.data)
                        if self.pullback_momentum == "pullback":
                            internal_momentum = self.optimizer.state[p][
                                "momentum_buffer"
                            ]
                            self.optimizer.state[p]["momentum_buffer"] = \
                                internal_momentum.mul_(self.la_alpha).add_(
                                    1.0 - self.la_alpha,
                                    param_state["cached_mom"]
                                )
                            param_state["cached_mom"] = \
                                self.optimizer.state[p]["momentum_buffer"]
                        elif self.pullback_momentum == "reset":
                            self.optimizer.state[p]["momentum_buffer"] = \
                                torch.zeros_like(p.data)

            return loss

TorchSolver = import_ctx.import_from('torch_solver', 'TorchSolver')


class Solver(TorchSolver):
    """Lookahead solver"""
    name = 'Lookahead-torch'

    # any parameter defined here is accessible as a class attribute
    parameters = {
        **TorchSolver.parameters,
        'lr': [1e-1],
        # parameters are taken from the appendix C.1 from the paper
        # https://arxiv.org/abs/1907.08610
        'weight_decay': [0.0, 5e-4],
        'momentum': [0.0, 0.9],
        'steps': [[3/10, 6/10, 8/10]],
        'gamma': [0.2],
        'la_steps': [5],
        'la_alpha': [0.8],
        'pullback_momentum': ['none'],
        'base_optimizer': ['sgd'],
    }

    base_optimizers_map = {
        'sgd': SGD,
        'adam': Adam,
    }

    def set_objective(self, **kwargs):
        super().set_objective(**kwargs)

        def optimizer_init(
            model_parameters,
            lr,
            weight_decay,
            momentum,
            **kwargs,
        ):
            base_optimizer_klass = self.base_optimizers_map[
                self.base_optimizer
            ]
            base_optimizer_kwargs = dict(lr=lr)
            if self.base_optimizer == 'sgd':
                base_optimizer_kwargs = dict(
                    **base_optimizer_kwargs,
                    weight_decay=weight_decay,
                    momentum=momentum,
                )
            base_optimizer = base_optimizer_klass(
                model_parameters,
                **base_optimizer_kwargs,
            )
            la_optimizer = Lookahead(base_optimizer, **kwargs)
            return la_optimizer

        self.optimizer_klass = optimizer_init
        self.optimizer_kwargs = dict(
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            la_steps=self.la_steps,
            la_alpha=self.la_alpha,
            pullback_momentum=self.pullback_momentum,
        )

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
        optimizer._backup_and_load_cache()
        while callback(model):
            optimizer._clear_and_load_backup()
            for X, y in tqdm(self.dataloader):
                if torch.cuda.is_available():
                    X, y = X.cuda(), y.cuda()
                optimizer.zero_grad()
                loss = criterion(model(X), y)
                loss.backward()

                optimizer.step()
            lr_schedule.step()
            optimizer._backup_and_load_cache()

        self.model = model
