from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import torch


def change_classification_head(large_model):
    """"Removes the big dense layers at the end of the large model
    and replaces them with a smaller one.
    """
    num_classes = large_model.classifier[-1].out_features
    dropout = large_model.classifier[-2].p
    large_model.classifier = torch.nn.Sequential(
        torch.nn.Linear(512, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=dropout),
        torch.nn.Linear(512, num_classes),
    )
    return large_model
