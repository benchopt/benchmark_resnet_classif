from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    import torch
    from torch.optim import SGD
    from tqdm import tqdm

    # Code from
    # https://github.com/facebookresearch/enorm/blob/master/enorm/enorm/enorm.py
    class ENorm:
        """
        Implements Equi-normalization for feedforward fully-connected and
        convolutional networks.
        Args:
            - named_params: the named parameters of your model, obtained as
            model.named_parameters()
            - optimizer: the optimizer, necessary for the momentum buffer update
            Note: only torch.optim.SGD supported for the moment
            - model_type: choose among ['linear', 'conv'] (see main.py)
            - c: asymmetric scaling factor that introduces a depth-wise penalty on
            the weights (default:1)
            - p: compute row and columns p-norms (default:2)
        Notes:
            - For all the supported architectures [fully connected, fully
            convolutional], we do not balance the last layer
            - In practice, we have found the training to be more stable when we do
            not balance the biases
        """

        def __init__(self, named_params, optimizer, model_type, c=1, p=2):
            self.named_params = list(named_params)
            self.optimizer = optimizer
            self.model_type = model_type
            self.momentum = self.optimizer.param_groups[0]['momentum']
            self.alpha = 0.5
            self.p = p

            # names to filter out
            to_remove = ['bn']
            filter_map = lambda x: not any(name in x[0] for name in to_remove)

            # weights and biases
            self.weights = [(n, p) for n, p in self.named_params if 'weight' in n]
            self.weights = list(filter(filter_map, self.weights))
            self.biases = []
            self.n_layers = len(self.weights)

            # scaling vector
            self.n_layers = len(self.weights)
            self.C = [c] * self.n_layers

        def _get_weight(self, i, orientation='l'):
            _, param = self.weights[i]
            if self.model_type != 'linear':
                if orientation == 'l':
                    # (C_in x k x k) x C_out
                    param = param.view(param.size(0), -1).t()
                else:
                    # C_in x (k x k x C_out)
                    param = param.permute(1, 2, 3, 0).contiguous().view(param.size(1), -1)
            return param

        def step(self):
            if self.model_type == 'linear':
                self._step_fc()
            else:
                self._step_conv()

        def _step_fc(self):
            left_norms = self._get_weight(0).norm(p=self.p, dim=1).data
            right_norms = self._get_weight(1).norm(p=self.p, dim=0).data

            for i in range(1, self.n_layers - 1):
                balancer = (right_norms / (left_norms * self.C[i - 1])).pow(self.alpha)

                left_norms = self._get_weight(i).norm(p=self.p, dim=1).data
                right_norms = self._get_weight(i + 1).norm(p=self.p, dim=0).data

                if len(self.biases) > 0: self.biases[i - 1][1].data.mul_(balancer)
                self._get_weight(i - 1).data.t_().mul_(balancer).t_()
                self._get_weight(i).data.mul_(1 / balancer)

        def _step_conv(self):
            left_w = self._get_weight(0, 'l')
            right_w = self._get_weight(1, 'r')

            left_norms = left_w.norm(p=2, dim=0).data
            right_norms = right_w.norm(p=2, dim=1).data

            for i in range(1, self.n_layers - 1):
                balancer = (right_norms / (left_norms * self.C[i-1])).pow(self.alpha)

                left_w = self._get_weight(i, 'l')
                right_w = self._get_weight(i + 1, 'r')

                left_norms = left_w.norm(p=2, dim=0).data
                right_norms = right_w.norm(p=2, dim=1).data

                self.weights[i - 1][1].data.mul_(
                    balancer.unsqueeze(1).unsqueeze(2).unsqueeze(3))
                self.weights[i][1].data.mul_(
                    1 / balancer.unsqueeze(1).unsqueeze(2).unsqueeze(0))

                if self.momentum:
                    self.optimizer.state[self.weights[i - 1][1]]['momentum_buffer'].mul_(
                        1 / balancer.unsqueeze(1).unsqueeze(2).unsqueeze(3))
                    self.optimizer.state[self.weights[i][1]]['momentum_buffer'].mul_(
                        balancer.unsqueeze(1).unsqueeze(2).unsqueeze(0))

    TorchSolver = import_ctx.import_from('torch_solver', 'TorchSolver')


class Solver(TorchSolver):
    """Lookahead solver"""
    name = 'Lookahead-torch'

    # any parameter defined here is accessible as a class attribute
    parameters = {
        **TorchSolver.parameters,
        'lr': [1e-1],
        'weight_decay': [0.0, 5e-4],
        'momentum': [0.0, 0.9],
        'c': [1],
        'p': [2],
    }

    def set_objective(self, **kwargs):
        super().set_objective(**kwargs)
        self.optimizer_klass = SGD
        self.optimizer_kwargs = dict(
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
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

        # enorm
        enorm = ENorm(
            model.named_parameters(),
            optimizer,
            c=self.c,
            p=self.p,
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
                enorm.step()
            lr_schedule.step()

        self.model = model
