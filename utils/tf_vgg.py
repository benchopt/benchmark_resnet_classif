from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from keras import layers
    from keras import models


def change_classification_head(large_model):
    """"Removes the big dense layers at the end of the large model
    and replaces them with a smaller one.
    """
    trimmed_model = models.Model(
        inputs=large_model.inputs,
        outputs=large_model.get_layer('flatten').output,
    )
    classes = large_model.output_shape[-1]
    small_dense = layers.Dense(512, activation='relu', name='fc1')
    dropout = layers.Dropout(0.5, name='dropout')
    classification_dense = layers.Dense(
        classes,
        activation='softmax',
        name='predictions',
    )
    small_model = models.Sequential([
        trimmed_model,
        small_dense,
        dropout,
        classification_dense,
    ])
    return small_model
