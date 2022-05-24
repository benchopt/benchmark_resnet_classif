import os

import numpy as np
import pytest
import tensorflow as tf
import torch

from benchopt.utils.safe_import import set_benchmark

set_benchmark('./')

CI = os.environ.get('CI', False)


def apply_torch_weights_to_tf(model, torch_weights_map):
    used_weights = []

    def apply_conv_torch_weights_to_tf(
        tf_layer,
        torch_layer_name,
        used_weights,
    ):
        weights_name = f'{torch_layer_name}.weight'
        torch_weights = torch_weights_map[weights_name]
        used_weights.append(weights_name)
        reshaped_torch_weights = np.transpose(
            torch_weights.detach().numpy(),
            (2, 3, 1, 0),
        )
        tf_layer.set_weights([reshaped_torch_weights])
        return used_weights
    for layer in model.layers + model.layers[-1].layers:
        weights = layer.get_weights()
        if weights:
            layer_ids = layer.name.split('_')
            layer_type = layer_ids[-1]
            if 'block' in layer.name:
                i_layer = str(int(layer_ids[0][4:]) - 1)
                i_block = str(int(layer_ids[1][5:]) - 1)
                i_conv = layer_ids[2]
                torch_layer_name = f'layer{i_layer}.{i_block}'
                if layer_type == 'conv':
                    if int(i_conv) > 0:
                        torch_layer_name += f'.conv{i_conv}'
                    else:
                        # downsampling layer
                        torch_layer_name += '.downsample.0'
                    used_weights = apply_conv_torch_weights_to_tf(
                        layer,
                        torch_layer_name,
                        used_weights,
                    )
                elif layer_type == 'bn':
                    continue
                else:
                    raise ValueError(f'Unknown layer type: {layer_type}')
            else:
                if layer_type == 'conv':
                    used_weights = apply_conv_torch_weights_to_tf(
                        layer,
                        'conv1',
                        used_weights,
                    )
                elif layer_type == 'bn' or layer_type == 'model':
                    continue
                elif layer_type == 'predictions':
                    torch_weights = torch_weights_map['fc.weight']
                    reshaped_torch_weights = np.transpose(
                        torch_weights.detach().numpy(),
                        (1, 0),
                    )
                    torch_bias = torch_weights_map['fc.bias']
                    layer.set_weights([
                        reshaped_torch_weights,
                        torch_bias.detach().numpy(),
                    ])
                    used_weights = used_weights + [
                        'fc.weight',
                        'fc.bias',
                    ]
                else:
                    try:
                        int(layer_type)
                    except ValueError:
                        raise ValueError(f'Unknown layer type: {layer_type}')
    weights_to_be_assigned = [
        w for w in torch_weights_map
        if 'tracked' not in w
        and 'bn' not in w
        and 'downsample.1' not in w
    ]
    print([w for w in weights_to_be_assigned if w not in used_weights])
    assert len(used_weights) == len(weights_to_be_assigned)


def generate_output_from_rand_image(
    framework,
    rand_image,
    torch_weights_map=None,
    optimizer=None,
    n_train_steps=1,
    inference_mode='train',
    batch_size=1,
    **extra_solver_kwargs,
):
    from datasets.cifar import Dataset
    from objective import Objective
    from solvers.adam_tf import Solver as TFAdamSolver
    from solvers.lookahead_tf import Solver as TFLookaheadSolver
    from solvers.sgd_tf import Solver as TFSGDSolver
    from solvers.adam_torch import Solver as TorchAdamSolver
    from solvers.lookahead_torch import Solver as TorchLookaheadSolver
    from solvers.sgd_torch import Solver as TorchSGDSolver
    from benchopt import safe_import_context
    with safe_import_context() as import_ctx:
        apply_coupled_weight_decay = import_ctx.import_from(
            'tf_helper', 'apply_coupled_weight_decay'
        )

    bench_dataset = Dataset.get_instance(framework=framework)
    bench_objective = Objective.get_instance(
        model_type='resnet',
        model_size='18',
    )
    bench_objective.set_dataset(bench_dataset)
    obj_dict = bench_objective.to_dict()
    model = obj_dict['model_init_fn']()
    lr = extra_solver_kwargs.pop('lr', 1e-3)
    if framework == 'pytorch':
        rand_image = torch.tensor(rand_image)

        if optimizer is not None:
            if optimizer == 'adam':
                solver_klass = TorchAdamSolver
            elif optimizer == 'lookahead':
                solver_klass = TorchLookaheadSolver
            elif optimizer == 'sgd':
                solver_klass = TorchSGDSolver

            def train_step(n_steps, model):
                model.train()
                solver = solver_klass.get_instance(
                    lr=lr,
                    **extra_solver_kwargs,
                )
                solver._set_objective(bench_objective)
                optimizer, _ = solver.set_lr_schedule_and_optimizer(model)
                criterion = torch.nn.CrossEntropyLoss()
                for _ in range(n_steps):
                    optimizer.zero_grad()
                    loss = criterion(
                        model(rand_image),
                        torch.ones(batch_size, dtype=torch.int64),
                    )
                    loss.backward()

                    optimizer.step()
                return model

        def model_fn(x):
            if inference_mode == 'train':
                model.train()
            else:
                model.eval()
            output = model(x)
            output = torch.softmax(output, dim=1)
            return output.detach().numpy()
        if torch.cuda.is_available():
            model = model.cuda()
            rand_image = rand_image.cuda()
        torch_weights_map = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
        }
    elif framework == 'tensorflow':
        rand_image = tf.convert_to_tensor(rand_image)

        def model_fn(x):
            output = model(x, training=inference_mode == 'train')
            return output.numpy()
        if torch_weights_map:
            apply_torch_weights_to_tf(model, torch_weights_map)

        if optimizer is not None:
            if optimizer == 'adam':
                solver_klass = TFAdamSolver
            elif optimizer == 'lookahead':
                solver_klass = TFLookaheadSolver
            elif optimizer == 'sgd':
                solver_klass = TFSGDSolver

            def train_step(n_steps, model):
                solver = solver_klass.get_instance(
                    lr=lr,
                    **extra_solver_kwargs,
                )
                solver._set_objective(bench_objective)
                _ = solver.get_lr_wd_cback(200)
                optimizer = solver.optimizer_klass(
                    weight_decay=solver.decoupled_wd*solver.lr,
                    **solver.optimizer_kwargs,
                )
                if solver.coupled_wd:
                    model = apply_coupled_weight_decay(
                        model,
                        solver.coupled_wd,
                    )
                model.compile(
                    optimizer=optimizer,
                    loss='sparse_categorical_crossentropy',
                )
                model.fit(
                    rand_image,
                    tf.ones([batch_size], dtype=tf.int64),
                    epochs=n_steps,
                    batch_size=batch_size,
                )
                return model
    if optimizer is not None:
        model = train_step(n_train_steps, model)
    output = model_fn(rand_image)
    return output, torch_weights_map


@pytest.mark.parametrize(
    'optimizer, extra_solver_kwargs', [
        (None, {}),
        ('adam', {}),
        ('lookahead', {}),
        ('sgd', {}),
        ('sgd', dict(momentum=0.9)),
        ('sgd', dict(weight_decay=5e-1)),
        ('sgd', dict(momentum=0.9, weight_decay=5e-1)),
        ('sgd', dict(momentum=0.9, weight_decay=5e-4, lr=1e-1)),
    ],
)
@pytest.mark.parametrize(
    'inference_mode', ['train', 'eval'],
)
def test_model_consistency(optimizer, extra_solver_kwargs, inference_mode):
    if optimizer == 'adam' and CI:
        pytest.skip('Adam is not yet aligned')
    if optimizer is not None and CI and inference_mode == 'eval':
        pytest.skip('eval tests not working because of batch norm discrepancy')
    np.random.seed(2)
    batch_size = 16
    rand_image = np.random.normal(
        size=(batch_size, 3, 32, 32),
    ).astype(np.float32)
    torch_output, torch_weights_map = generate_output_from_rand_image(
        'pytorch',
        rand_image,
        optimizer=optimizer,
        **extra_solver_kwargs,
        n_train_steps=2,
        inference_mode=inference_mode,
        batch_size=batch_size,
    )
    rand_image = np.transpose(rand_image, (0, 2, 3, 1))
    if 'weight_decay' in extra_solver_kwargs:
        extra_solver_kwargs['coupled_weight_decay'] = extra_solver_kwargs.pop(
            'weight_decay',
        )
    tf_output, _ = generate_output_from_rand_image(
        'tensorflow',
        rand_image,
        torch_weights_map,
        optimizer=optimizer,
        **extra_solver_kwargs,
        n_train_steps=2,
        inference_mode=inference_mode,
        batch_size=batch_size,
    )
    np.testing.assert_allclose(
        torch_output,
        tf_output,
        rtol=1e-4,
        atol=1e-5,
    )
