from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import pytorchcv.models.wrn_cifar as wrn


WRN_map = {
    '40_8': wrn.wrn40_8_cifar10,
    '28_10': wrn.wrn28_10_cifar10,
    '16_10': wrn.wrn16_10_cifar10,
}


def get_wrn_klass(model_size):
    def wrn(num_classes=10):
        model = WRN_map[model_size](
            classes=num_classes,
        )
        return model
    return wrn
