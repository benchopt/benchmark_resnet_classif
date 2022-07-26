import tensorflow as tf

from benchopt.utils.safe_import import set_benchmark

set_benchmark('./')


def test_random_resized_crop():
    from benchopt import safe_import_context
    with safe_import_context() as import_ctx:
        RandomResizedCrop = import_ctx.import_from(
            'tf_helper', 'RandomResizedCrop'
        )
    images = tf.random.normal([1, 32, 32, 3])
    crops = RandomResizedCrop(
        scale=(0.08, 1.0),
        ratio=(0.75, 1.33),
        crop_shape=(16, 16),
    )(images)
    assert crops.shape == (1, 16, 16, 3)
