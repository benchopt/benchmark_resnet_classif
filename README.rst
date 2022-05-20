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

	$ benchopt run benchmark_resnet_classif -o "*18" -d "cifar[*random_state=42*with_validation=False]" -s "adam-torch[batch_size=128,coupled_weight_decay=0.0,data_aug=True,decoupled_weight_decay=0.02,*,lr_schedule=cosine]"  --max-runs 200 --n-repetitions 1

Use `benchopt run -h` for more details about these options, or visit https://benchopt.github.io/api.html.

Extension
---------

If you want to add a new solver, you will need probably need to inherit one of the base solver classes from PyTorch, TensorFlow or PyTorch-Lightning.
For example, to implement a new PyTorch-based solver, you will need at the beginning of your solver class to add the following:

::


   from benchopt import safe_import_context


   with safe_import_context() as import_ctx:
      from torch.optim import ...

   TorchSolver = import_ctx.import_from('torch_solver', 'TorchSolver')

   class Solver(TorchSolver):
      ...





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
