Benchmark repository for ResNet fitting on classification
=====================
|Build Status| |Python 3.6+| |TensorFlow 2.8+| |PyTorch 1.10+| |PyTorch-Lightning 1.6+|

BenchOpt is a package to simplify and make more transparent and
reproducible the comparisons of optimization algorithms.
This benchmark is dedicated to solver of the ResNet classification fitting problem:

.. math::

    \min_{w} \sum_i L(f_w(x_i), y_i)

where :math:`i` is the sample index, :math:`x_i` is the input image, :math:`y_i` is the sample label, and :math:`L` is the cross-entropy loss function.


Use
--------

This benchmark can be run using the following commands:

.. code-block::

   $ pip install -U benchopt
   $ git clone https://github.com/benchopt/benchmark_resnet_classif
   $ benchopt run benchmark_resnet_classif

While this command would run the entire benchmark, which includes several models, datasets and solvers, sequentially,
you can restrict the run to a specific model, dataset and/or solver by passing the corresponding arguments.
For example if I want to run the benchmark for the ResNet18 model on CIFAR10 dataset with the Adam solver, without a validation set, and for a single random seed:

.. code-block::

	$ benchopt run . -o "*[model_size=18]" -d "cifar[random_state=42,with_validation=False]" -s "adam-torch[batch_size=128,coupled_weight_decay=0.0,data_aug=True,decoupled_weight_decay=0.02,lr_schedule=cosine]"  --max-runs 200 --n-repetitions 1

Use `benchopt run -h` for more details about these options, or visit https://benchopt.github.io/api.html.

Extension
---------

If you want to add a new solver, you need to inherit one of the base solver classes from PyTorch, TensorFlow or PyTorch-Lightning.
For example, to implement a new PyTorch-based solver with the Adam optimizer, you can add the following python file in the `solvers <solvers>`_ folder:

.. code:: python

   from benchopt import BaseSolver, safe_import_context

   with safe_import_context() as import_ctx:
      import torch


   class Solver(BaseSolver):

      parameters = {
         'lr': [1e-3],
         'batch_size': [128],
      }

      def skip(self, framework, **_kwargs,):
         if framework != 'pytorch':
            return True, 'Not a torch dataset/objective'
         return False, None

      def set_objective(self, dataset, model_init_fn, **_kwargs):
         self.model_init_fn = model_init_fn
         self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=10,
            persistent_workers=True,
            pin_memory=True,
            shuffle=True,
         )

      @staticmethod
      def get_next(stop_val):
         # evaluate the model at every epoch.
         return stop_val + 1

      def run(self, callback):
         # model weight initialization
         self.model = self.model_init_fn()
         criterion = torch.nn.CrossEntropyLoss()

         max_epochs = callback.stopping_criterion.max_runs
         optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
         # Initial evaluation
         while callback(self.model):
            for X, y in self.dataloader:
                  if torch.cuda.is_available():
                     X, y = X.cuda(), y.cuda()
                  optimizer.zero_grad()
                  loss = criterion(self.model(X), y)
                  loss.backward()

                  optimizer.step()

      def get_result(self):
         return dict(model=self.model)

If you want to use a more complex solver, using a learning rate scheduler, as well as data augmentation,
you can subclass the `TorchSolver <utils/torch_solver.py>`_ class we provide:

.. code:: python

   from benchopt import safe_import_context

   from benchmark_utils.torch_solver import TorchSolver

   with safe_import_context() as import_ctx:
      from torch.optim import Adam


   class Solver(TorchSolver):
      """Adam solver"""
      name = 'Adam-torch'

      # any parameter defined here is accessible as a class attribute
      parameters = {
         **TorchSolver.parameters,
         'lr': [1e-3],
         'weight_decay': [0.0, 5e-4],
      }

      def set_objective(self, **kwargs):
         super().set_objective(**kwargs)
         self.optimizer_klass = Adam
         self.optimizer_kwargs = dict(
               lr=self.lr,
               weight_decay=self.weight_decay,
         )

If you want to modify the data augmentation policy you will have to override the :code:`set_objective` function.
If you want to use a different learning rate scheduler, you will have to override the :code:`set_lr_schedule_and_optimizer` function.
We are in the process of making these functions more modular to enable easier customization.



.. |Build Status| image:: https://github.com/benchopt/benchmark_resnet_classif/workflows/Tests/badge.svg
   :target: https://github.com/benchopt/benchmark_resnet_classif/actions
.. |Python 3.6+| image:: https://img.shields.io/badge/python-3.6%2B-blue
   :target: https://www.python.org/downloads/release/python-360/
.. |TensorFlow 2.8+| image:: https://img.shields.io/badge/TensorFlow-2.8%2B-orange
   :target: https://www.tensorflow.org/?hl=fr
.. |PyTorch 1.10+| image:: https://img.shields.io/badge/PyTorch-1.10%2B-red
   :target: https://pytorch.org/
.. |PyTorch-Lightning 1.6+| image:: https://img.shields.io/badge/PyTorch--Lightning-1.6%2B-blueviolet
   :target: https://pytorch-lightning.readthedocs.io/en/latest/
