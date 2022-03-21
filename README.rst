Benchmark repository for ResNet fitting on classification
=====================
|Build Status| |Python 3.6+|

BenchOpt is a package to simplify and make more transparent and
reproducible the comparisons of optimization algorithms.
This benchmark is dedicated to solver of the ResNet classification fitting problem:

.. math::

    \min_{w} \sum_i L(f_w(x_i), y_i)

where :math:`i` is the sample index, :math:`x_i` is the input image, :math:`y_i` is the sample label, and :math:`L` is the cross-entropy loss function.


Install
--------

This benchmark can be run using the following commands:

.. code-block::

   $ pip install -U benchopt
   $ git clone https://github.com/zaccharieramzi/benchmark_resnet_classif
   $ benchopt run benchmark_resnet_classif

Apart from the problem, options can be passed to `benchopt run`, to restrict the benchmarks to some solvers or datasets, e.g.:

.. code-block::

	$ benchopt run benchmark_resnet_classif -s solver1 -d dataset2 --max-runs 10 --n-repetitions 10

Current workaround for https://github.com/benchopt/benchopt/issues/306:

..code-block::

   $ benchopt install -e .
   $ conda activate benchopt_benchmark_resnet_classif
   $ conda install pytorch torchvision cpuonly -c pytorch
   $ conda install -c conda-forge pytorch-lightning



Use `benchopt run -h` for more details about these options, or visit https://benchopt.github.io/api.html.

.. |Build Status| image:: https://github.com/zaccharieramzi/benchmark_resnet_classif/workflows/Tests/badge.svg
   :target: https://github.com/zaccharieramzi/benchmark_resnet_classif/actions
.. |Python 3.6+| image:: https://img.shields.io/badge/python-3.6%2B-blue
   :target: https://www.python.org/downloads/release/python-360/
