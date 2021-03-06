from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from keras.applications.resnet import ResNet
    from keras import initializers
    from keras import layers
    from keras import models


# We might consider other options like
# https://github.com/keras-team/keras-contrib
# But it looks dead, and not moved to tf-addons
# Same for https://github.com/qubvel/classification_models

# XXX: make sure to remove this once this PR is merged:
# https://github.com/keras-team/keras/pull/16363

# Another option would be to use tensorflow/models once this PR is merged:
# https://github.com/tensorflow/models/pull/10584

def basic_block(x, filters, stride=1, use_bias=True, conv_shortcut=True,
                name=None):
    """A basic residual block for ResNet18 and 34.

    Args:
    x: input tensor.
    filters: integer, filters of the bottleneck layer.
    kernel_size: default 3, kernel size of the bottleneck layer.
    stride: default 1, stride of the first layer.
    conv_shortcut: default True, use convolution shortcut if True,
        otherwise identity shortcut.
    name: string, block label.

    Returns:
    Output tensor for the basic residual block.
    """
    bn_axis = 3
    kernel_size = 3
    torch_init = initializers.VarianceScaling(
        scale=2.0,
        mode='fan_out',
        distribution='untruncated_normal',
    )

    if conv_shortcut:
        shortcut = layers.Conv2D(
            filters,
            1,
            strides=stride,
            use_bias=use_bias,
            name=name + '_0_conv',
            kernel_initializer=torch_init,
        )(x)
        shortcut = layers.BatchNormalization(
            momentum=0.9,
            axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    if stride > 1:
        x = layers.ZeroPadding2D(
            padding=((1, 0), (1, 0)),
            name=name + '_1_pad',
        )(x)
        padding_mode = 'valid'
    else:
        padding_mode = 'same'
    x = layers.Conv2D(
        filters, kernel_size, padding=padding_mode, strides=stride,
        kernel_initializer=torch_init,
        use_bias=use_bias,
        name=name + '_1_conv')(x)
    x = layers.BatchNormalization(
        momentum=0.9,
        axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.Conv2D(
        filters,
        kernel_size,
        padding='SAME',
        use_bias=use_bias,
        kernel_initializer=torch_init,
        name=name + '_2_conv',
    )(x)
    x = layers.BatchNormalization(
        momentum=0.9,
        axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)

    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)
    return x


def bottleneck_block(x, filters, kernel_size=3, stride=1, conv_shortcut=True,
                     use_bias=True, name=None):
    """A residual block with a bottle neck used in ResNet 50, 101, 152.

    Args:
    x: input tensor.
    filters: integer, filters of the bottleneck layer.
    kernel_size: default 3, kernel size of the bottleneck layer.
    stride: default 1, stride of the first layer.
    conv_shortcut: default True, use convolution shortcut if True,
        otherwise identity shortcut.
    name: string, block label.

    Returns:
    Output tensor for the residual block.
    """
    bn_axis = 3
    torch_init = initializers.VarianceScaling(
        scale=2.0,
        mode='fan_out',
        distribution='untruncated_normal',
    )

    if conv_shortcut:
        shortcut = layers.Conv2D(
            4 * filters,
            1,
            strides=stride,
            use_bias=use_bias,
            kernel_initializer=torch_init,
            name=name + '_0_conv',
            )(x)
        shortcut = layers.BatchNormalization(
            momentum=0.9,
            axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(
        filters,
        1,
        strides=stride,
        use_bias=use_bias,
        kernel_initializer=torch_init,
        name=name + '_1_conv',
    )(x)
    x = layers.BatchNormalization(
        momentum=0.9,
        axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.Conv2D(
        filters,
        kernel_size,
        padding='SAME',
        use_bias=use_bias,
        kernel_initializer=torch_init,
        name=name + '_2_conv',
    )(x)
    x = layers.BatchNormalization(
        momentum=0.9,
        axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)

    x = layers.Conv2D(
        4 * filters,
        1,
        use_bias=use_bias,
        kernel_initializer=torch_init,
        name=name + '_3_conv',
    )(x)
    x = layers.BatchNormalization(
        momentum=0.9,
        axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn')(x)

    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)
    return x


def stack_block(
    x,
    filters,
    n_blocks,
    block_fn,
    stride1=2,
    first_shortcut=True,
    name=None,
    use_bias=False,
):
    """A set of stacked residual blocks.

    Args:
    x: input tensor.
    filters: integer, filters of the bottleneck layer in a block.
    n_blocks: integer, blocks in the stacked blocks.
    block_fn: callable, function defining one block.
    stride1: default 2, stride of the first layer in the first block.
    name: string, stack label.

    Returns:
    Output tensor for the stacked basic blocks.
    """
    x = block_fn(
        x,
        filters,
        stride=stride1,
        conv_shortcut=first_shortcut,
        use_bias=use_bias,
        name=name + '_block1',
    )
    for i in range(2, n_blocks + 1):
        x = block_fn(
            x,
            filters,
            conv_shortcut=False,
            use_bias=use_bias,
            name=name + '_block' + str(i),
        )
    return x


def remove_initial_downsample(large_model, use_bias=False):
    torch_init = initializers.VarianceScaling(
        scale=2.0,
        mode='fan_out',
        distribution='untruncated_normal',
    )
    trimmed_model = models.Model(
        inputs=large_model.get_layer('conv2_block1_1_conv').input,
        outputs=large_model.outputs,
    )
    first_conv = layers.Conv2D(
        64,
        3,
        activation='linear',
        padding='same',
        use_bias=use_bias,
        kernel_initializer=torch_init,
        name='conv1_conv',
    )
    input_shape = list(large_model.input_shape[1:])
    input_shape[0] = input_shape[0] // 4
    input_shape[1] = input_shape[1] // 4
    small_model = models.Sequential([
        layers.Input(input_shape),
        first_conv,
        layers.BatchNormalization(
            momentum=0.9,
            axis=-1,
            epsilon=1.001e-5,
            name='conv1_bn',
        ),
        layers.Activation('relu', name='conv1_relu'),
        trimmed_model,
    ])
    return small_model


def change_dense_init(model):
    torch_init = initializers.VarianceScaling(
        scale=1.0,
        mode='fan_in',
        distribution='uniform',
    )
    for layer in model.layers:
        if isinstance(layer, layers.Dense):
            layer.kernel_initializer = torch_init
            layer.bias_initializer = torch_init
            layer.build(layer.input_spec.shape)


def ResNet18(include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000,
             use_bias=True,
             no_initial_downsample=False,
             dense_init='tf',
             **kwargs):
    """Instantiates the ResNet18 architecture."""

    def stack_fn(x):
        x = stack_block(
            x,
            64,
            2,
            basic_block,
            use_bias=use_bias,
            first_shortcut=False,
            stride1=1,
            name='conv2',
        )
        x = stack_block(
            x,
            128,
            2,
            basic_block,
            use_bias=use_bias,
            name='conv3',
        )
        x = stack_block(
            x,
            256,
            2,
            basic_block,
            use_bias=use_bias,
            name='conv4',
        )
        return stack_block(
            x,
            512,
            2,
            basic_block,
            use_bias=use_bias,
            name='conv5',
        )

    model = ResNet(
        stack_fn,
        False,
        use_bias,
        'resnet18',
        include_top,
        weights,
        input_tensor,
        input_shape,
        pooling,
        classes,
        **kwargs,
    )
    if no_initial_downsample:
        model = remove_initial_downsample(model, use_bias=use_bias)
    if dense_init == 'torch':
        change_dense_init(model)
    return model


def ResNet34(include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000,
             use_bias=True,
             no_initial_downsample=False,
             dense_init='tf',
             **kwargs):
    """Instantiates the ResNet34 architecture."""

    def stack_fn(x):
        x = stack_block(
            x,
            64,
            3,
            basic_block,
            use_bias=use_bias,
            first_shortcut=False,
            stride1=1,
            name='conv2',
        )
        x = stack_block(
            x,
            128,
            4,
            basic_block,
            use_bias=use_bias,
            name='conv3',
        )
        x = stack_block(
            x,
            256,
            6,
            basic_block,
            use_bias=use_bias,
            name='conv4',
        )
        return stack_block(
            x,
            512,
            3,
            basic_block,
            use_bias=use_bias,
            name='conv5',
        )

    model = ResNet(
        stack_fn,
        False,
        use_bias,
        'resnet34',
        include_top,
        weights,
        input_tensor,
        input_shape,
        pooling,
        classes,
        **kwargs,
    )
    if no_initial_downsample:
        model = remove_initial_downsample(model, use_bias=use_bias)
    if dense_init == 'torch':
        change_dense_init(model)
    return model


def ResNet50(include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000,
             use_bias=True,
             no_initial_downsample=False,
             dense_init='tf',
             **kwargs):
    """Instantiates the ResNet50 architecture."""

    def stack_fn(x):
        x = stack_block(
            x,
            64,
            3,
            bottleneck_block,
            use_bias=use_bias,
            stride1=1,
            name='conv2',
        )
        x = stack_block(
            x,
            128,
            4,
            bottleneck_block,
            use_bias=use_bias,
            name='conv3',
        )
        x = stack_block(
            x,
            256,
            6,
            bottleneck_block,
            use_bias=use_bias,
            name='conv4',
        )
        return stack_block(
            x,
            512,
            3,
            bottleneck_block,
            use_bias=use_bias,
            name='conv5',
        )

    model = ResNet(
        stack_fn,
        False,
        use_bias,
        'resnet50',
        include_top,
        weights,
        input_tensor,
        input_shape,
        pooling,
        classes,
        **kwargs,
    )
    if no_initial_downsample:
        model = remove_initial_downsample(model, use_bias=use_bias)
    if dense_init == 'torch':
        change_dense_init(model)
    return model
