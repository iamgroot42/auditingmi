from mib.models.wide_resnet import Wide_ResNet
from mib.models.nasnet import NasNetA
from mib.models.shufflenetv2 import ShuffleNetV2
from mib.models.mlp import MLP, MLPQuadLoss
from mib.models.cnn import CNN
from torch import nn


MODEL_MAPPING = {
    "wide_resnet_28_1": {
        # 4.8s/it
        # 100e - 88%, 0.45
        # 200e - 90%, 0.38
        # 500e - 90%, ~0.35
        "model": (Wide_ResNet, (28, 1)),
        "criterion": nn.CrossEntropyLoss(),
        "hparams": {"batch_size": 256, "learning_rate": 0.1, "epochs": 500},
    },
    "wide_resnet_28_2": {
        # 500e - 92%
        # 7s/it
        "model": (Wide_ResNet, (28, 2)),
        "criterion": nn.CrossEntropyLoss(),
        "hparams": {"batch_size": 256, "learning_rate": 0.1, "epochs": 100},
    },
    "wide_resnet_28_10": {
        "model": (Wide_ResNet, (28, 10)),
        "criterion": nn.CrossEntropyLoss(),
        "hparams": {"batch_size": 256, "learning_rate": 0.1, "epochs": 100},
    },
    "nasnet": {
        "model": (NasNetA, (4, 2, 44, 44)),
        "criterion": nn.CrossEntropyLoss(),
        "hparams": {"batch_size": 256, "learning_rate": 0.1, "epochs": 100},
    },
    "shufflenet_v2_s": {
        "model": (ShuffleNetV2, (1,)),
        "criterion": nn.CrossEntropyLoss(),
        "hparams": {"batch_size": 256, "learning_rate": 0.1, "epochs": 100},
    },
    "shufflenet_v2_m": {
        "model": (ShuffleNetV2, (1.5,)),
        "criterion": nn.CrossEntropyLoss(),
        "hparams": {"batch_size": 256, "learning_rate": 0.1, "epochs": 100},
    },
    "mlp4": {
        "model": (
            MLP,
            (
                600,
                [512, 256, 128, 64],
            ),
        ),
        "criterion": nn.CrossEntropyLoss(),
        # 100e -0.54, 82%
        # 200e- 0.52, 82%
        # 500e, 0.1 LR _ 0.44, 84%
        # 400e, 0.1 LR - 0.44, ~84%
        # 300e, 0.1 LR - 0.45, ~83%
        # 200e, 0.1 LR - 0.47, 83%
        # 100e, 0.1 LR -  0.49, 83%
        "hparams": {"batch_size": 256, "learning_rate": 0.1, "epochs": 400},
    },
    "mlp4_p100s": {
        "model": (
            MLP,
            (
                600,
                [512, 256, 128, 64],
            ),  
        ),
        "criterion": nn.CrossEntropyLoss(),
        "hparams": {"batch_size": 256, "learning_rate": 0.01, "epochs": 200},
    },
    "mlp3": {
        "model": (
            MLP,
            (
                600,
                [128, 64, 32],
            ),
        ),
        "criterion": nn.CrossEntropyLoss(),
        "hparams": {"batch_size": 256, "learning_rate": 0.05, "epochs": 120},
    },
    "mlp3_small": {
        "model": (
            MLP,
            (
                600,
                [32, 32, 8],
            ),
        ),
        "criterion": nn.CrossEntropyLoss(),
        "hparams": {"batch_size": 256, "learning_rate": 0.02, "epochs": 120},
    },
    "mlp2": {
        "model": (
            MLP,
            (
                600,
                [32],
            ),
        ),
        "criterion": nn.CrossEntropyLoss(),
        "hparams": {"batch_size": 256, "learning_rate": 0.02, "epochs": 120},
    },
    "mlp2_fmnist": {
        "model": (
            MLP,
            (
                784,
                [6],
            ),
        ),
        "criterion": nn.CrossEntropyLoss(),
        "hparams": {"batch_size": 256, "learning_rate": 0.01, "epochs": 120},
    },
    "lr": {
        "model": (
            MLP,
            (
                600,
                [],
            ),
        ),
        "criterion": nn.CrossEntropyLoss(),
        "hparams": {"batch_size": 256, "learning_rate": 0.01, "epochs": 120},
    },
    "mlp_mnist17": {
        "model": (
            MLP,
            (
                784,
                [4],
            ),
        ),
        "criterion": nn.BCEWithLogitsLoss(),
        "hparams": {"batch_size": 128, "learning_rate": 0.001, "epochs": 100},
    },
    "mlp_mnistodd": {
        "model": (
            MLP,
            (
                784,
                [8],
            ),
        ),
        "criterion": nn.BCEWithLogitsLoss(),
        "hparams": {"batch_size": 128, "learning_rate": 0.01, "epochs": 100},
    },
    "mlp_mnistodd_mse": {
        "model": (MLPQuadLoss, (784, [8])),
        "criterion": nn.MSELoss(),
        "hparams": {"batch_size": 128, "learning_rate": 0.01, "epochs": 100},
    },
    "lr_mse": {
        "model": (MLPQuadLoss, (784, [])),
        "criterion": nn.MSELoss(),
        "hparams": {"batch_size": 128, "learning_rate": 0.01, "epochs": 80},
    },
    "cnn32_3_max": {
        # 4.7s/it
        "model": (CNN, (3, 32, "max", 3)),
        "criterion": nn.CrossEntropyLoss(),
        # 500e - 
        # 200e - 
        # 100e - 
        "hparams": {"batch_size": 256, "learning_rate": 0.1, "epochs": 100},
    },
    "cnn32_3_avg": {
        "model": (CNN, (3, 32, "avg", 3)),
        "criterion": nn.CrossEntropyLoss(),
        # 200e - 87%, 0.67
        # 100e - 86%, 0.76
        "hparams": {"batch_size": 256, "learning_rate": 0.1, "epochs": 100},
    },
    "cnn64_3_max": {
        "model": (CNN, (3, 64, "max", 3)),
        "criterion": nn.CrossEntropyLoss(),
        # 5s/it
        # 500e -
        # 200e - 89%, 0.53
        # 100e - 88%, 0.59
        "hparams": {"batch_size": 256, "learning_rate": 0.1, "epochs": 500},
    },
    "cnn64_3_avg": {
        "model": (CNN, (3, 64, "avg", 3)),
        "criterion": nn.CrossEntropyLoss(),
        # 200e - 87%, 0.63
        # 100e - 87%, 0.73
        "hparams": {"batch_size": 256, "learning_rate": 0.1, "epochs": 100},
    },
    # "mlp4_slow": {
    #     "model": (MLP, ([512, 256, 128, 64], )),
    #     "criterion": nn.CrossEntropyLoss(),
    #     "hparams": {"batch_size": 256, "learning_rate": 0.01, "epochs": 50},
    # }
    # "efficientnet_v2_s": {
    #     "model": (efficientnet_v2_s, ()),
    #     "criterion": nn.CrossEntropyLoss(),
    #     "hparams": {"batch_size": 256, "learning_rate": 0.1, "epochs": 100},
    # },
}


def get_model(name: str, n_classes: int):
    if name not in MODEL_MAPPING:
        raise ValueError(f"Model {name} not found.")
    model_info = MODEL_MAPPING[name]
    model_class, params = model_info["model"]
    criterion = model_info["criterion"]
    hparams = model_info["hparams"]
    return model_class(*params, n_classes), criterion, hparams
