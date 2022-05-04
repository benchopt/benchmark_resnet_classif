import numpy as np
import tensorflow as tf
import torch

from benchopt.utils.safe_import import set_benchmark

set_benchmark('./')




def generate_output_from_rand_image(framework, rand_image, torch_weights_map=None):
    from datasets.cifar import Dataset
    from objective import Objective

    bench_dataset = Dataset.get_instance(framework=framework)
    bench_objective = Objective.get_instance(model_type='resnet', model_size='18')
    bench_objective.set_dataset(bench_dataset)
    obj_dict = bench_objective.to_dict()
    model = obj_dict['model_init_fn']()
    if framework == 'pytorch':
        model.eval()
        rand_image = torch.tensor(rand_image)

        def model_fn(image):
            with torch.no_grad():
                output = model(image)
                output = torch.softmax(output, dim=1)
                return output.numpy()
        if torch.cuda.is_available():
            model = model.cuda()
            rand_image = rand_image.cuda()
        torch_weights_map = {
            name: param
            for name, param in model.named_parameters()
        }
        from torchsummary import summary
        summary(model)
    elif framework == 'tensorflow':
        rand_image = tf.convert_to_tensor(rand_image)
        model_fn = model.predict
        if torch_weights_map:
            used_weights = []
            def apply_conv_torch_weights_to_tf(tf_layer, torch_layer_name, used_weights):
                weights_name = f'{torch_layer_name}.weight'
                torch_weights = torch_weights_map[weights_name]
                used_weights.append(weights_name)
                reshaped_torch_weights = np.transpose(torch_weights.detach().numpy(), (2, 3, 1, 0))
                tf_layer.set_weights([reshaped_torch_weights])
                return used_weights
            for layer in model.layers + model.layers[-1].layers:
                weights = layer.get_weights()
                if weights:
                    layer_ids = layer.name.split('_')
                    layer_type = layer_ids[-1]
                    if 'block' in layer.name:
                        i_layer = str(int(layer_ids[0][4:]) -1)
                        i_block = str(int(layer_ids[1][5:]) - 1)
                        i_conv = layer_ids[2]
                        torch_layer_name = f'layer{i_layer}.{i_block}'
                        if layer_type == 'conv':
                            if int(i_conv) > 0:
                                torch_layer_name += f'.conv{i_conv}'
                            else:
                                # downsampling layer
                                torch_layer_name += '.downsample.0'
                            used_weights = apply_conv_torch_weights_to_tf(layer, torch_layer_name, used_weights)
                        elif layer_type == 'bn':
                            continue
                        else:
                            raise ValueError(f'Unknown layer type: {layer_type}')
                    else:
                        if layer_type == 'conv':
                            used_weights = apply_conv_torch_weights_to_tf(layer, 'conv1', used_weights)
                        elif layer_type == 'bn' or layer_type == 'model':
                            continue
                        elif layer_type == 'predictions':
                            torch_weights = torch_weights_map['fc.weight']
                            reshaped_torch_weights = np.transpose(torch_weights.detach().numpy(), (1, 0))
                            torch_bias = torch_weights_map['fc.bias']
                            layer.set_weights([reshaped_torch_weights, torch_bias.detach().numpy()])
                            used_weights = used_weights + ['fc.weight', 'fc.bias']
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
    output = model_fn(rand_image)
    return output, torch_weights_map


rand_image = np.random.normal(size=(1, 3, 32, 32)).astype(np.float32)
torch_output, torch_weights_map = generate_output_from_rand_image('pytorch', rand_image)
rand_image = np.transpose(rand_image, (0, 2, 3, 1))
tf_output, _ = generate_output_from_rand_image('tensorflow', rand_image, torch_weights_map)
np.testing.assert_allclose(torch_output, tf_output)
