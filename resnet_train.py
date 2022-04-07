import hydra
import tensorflow as tf
import tfds

from utils.tf_resnets import ResNet18, ResNet34


@hydra.main(config_path="config.yaml")
def main(cfg):
    # train a resnet on imagenet
    if cfg.model_id == "18":
        model_klass = ResNet18
    elif cfg.model_id == "34":
        model_klass = ResNet34
    else:
        raise ValueError("model_id must be 18 or 34")
    # instantiate a tf Dataset with the imagenet dataset
    imagenet_ds =

if __name__ == "__main__":
    main()
