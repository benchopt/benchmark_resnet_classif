import time

import torch

from benchopt.utils.safe_import import set_benchmark

set_benchmark('./')

from datasets.simulated import Dataset
from objective import Objective


def profile(framework, n_runs=100):
    bench_dataset = Dataset.get_instance(framework=framework)
    bench_objective = Objective.get_instance(model_type='resnet', model_size='18')
    bench_objective.set_dataset(bench_dataset)
    obj_dict = bench_objective.to_dict()
    model = obj_dict['model_init_fn']()
    dataset = obj_dict['dataset']
    if framework == 'pytorch':
        image, _ = dataset[0]
        if torch.cuda.is_available():
            image = image.cuda()
    elif framework == 'tensorflow':
        image, _ = next(iter(dataset))
    # warm-up
    for _ in range(5):
        model(image)
    # timing
    start = time.time()
    for _ in range(n_runs):
        model(image)
    end = time.time()
    return end - start


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--framework', default='pytorch', choices=['pytorch', 'tensorflow'])
    parser.add_argument('--n_runs', default=100, type=int)
    args = parser.parse_args()
    timing = profile(args.framework, args.n_runs)
    print(f'{args.framework} {timing / args.n_runs}')
