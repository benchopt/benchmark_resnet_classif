from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import tensorflow as tf
    import tf2cv.models.wrn_cifar as wrn


WRN_map = {
    '40-8': wrn.wrn40_8_cifar10,
    '28-10': wrn.wrn28_10_cifar10,
    '16-10': wrn.wrn16_10_cifar10,
}


def get_wrn_klass(model_size):
    def wrn(
        num_classes=10,
        input_shape=(32, 32, 3),
        classifier_activation='softmax',
        **dummy_kwargs,
    ):
        model = WRN_map[model_size](
            classes=num_classes,
            in_channels=input_shape[-1],
            in_size=input_shape[:1],
        )
        model.output1.activation = tf.keras.layers.Activation(
            classifier_activation,
        )
        return model
    return wrn
