import time

import torch
from torchsummary import summary

from benchopt.utils.safe_import import set_benchmark

set_benchmark('./')


def profile(framework, n_runs=100, verbose=1):
    from datasets.simulated import Dataset
    from objective import Objective

    bench_dataset = Dataset.get_instance(framework=framework)
    bench_objective = Objective.get_instance(
        model_type='resnet',
        model_size='18',
    )
    bench_objective.set_dataset(bench_dataset)
    obj_dict = bench_objective.to_dict()
    model = obj_dict['model_init_fn']()
    dataset = obj_dict['dataset']
    if framework == 'pytorch':
        model.train()
        # summary of torch model
        if verbose > 0:
            summary(model)

        def model_fn(image):
            output = model(image)
            output = torch.softmax(output, dim=1)
            if torch.cuda.is_available():
                return output.detach().cpu().numpy()
            else:
                return output.detach().numpy()
        image, _ = dataset[0]
        if torch.cuda.is_available():
            model = model.cuda()
            image = image.cuda()
        image = image.unsqueeze(0)
    elif framework == 'tensorflow':
        if verbose > 0:
            model.summary()
        image, _ = next(iter(dataset))
        image = image[None]

        def model_fn(image):
            return model(image).numpy()
    # warm-up
    for _ in range(5):
        model_fn(image)
    # timing
    start = time.time()
    for _ in range(n_runs):
        model_fn(image)
    end = time.time()
    return end - start


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--framework',
        default='pytorch',
        choices=['pytorch', 'tensorflow'],
    )
    parser.add_argument('--n_runs', default=100, type=int)
    args = parser.parse_args()
    timing = profile(args.framework, args.n_runs)
    print(f'{args.framework} {timing / args.n_runs}')
