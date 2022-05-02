from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import torch


def remove_initial_downsample(large_model):
    large_model.conv1 = torch.nn.Conv2d(
        3,
        64,
        kernel_size=3,
        stride=1,
        padding=3,
        bias=False,
    )
    large_model.maxpool = torch.nn.Identity()
    return large_model
