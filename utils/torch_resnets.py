from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import torch
    from torchvision.models import ResNet


def remove_initial_downsample(large_model):
    large_model.conv1 = torch.nn.Conv2d(
        3,
        64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    large_model.maxpool = torch.nn.Identity()
    return large_model


def wide_resnet(n_layers, widening_factor):
    layers_map = {
        16: [1, ]

    }
    model = ResNet(
        ResNet.BasicBlock,
        [n_layers, n_layers, n_layers],

    )
    return model
