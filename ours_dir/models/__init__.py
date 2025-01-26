from .mobilenetv2 import mobile_half
from .mobilenetv2_imagenet import mobilenet_v2
from .resnet import (resnet8, resnet8x4, resnet8x4_double, resnet14, resnet20,
                     resnet32, resnet32x4, resnet44, resnet56, resnet110)
from .resnetv2 import (resnet18, resnet18x2, resnet34, resnet34x4,
                       resnext50_32x4d, wide_resnet50_2)
from .resnetv2_org import ResNet50, ResNet18
from .ShuffleNetv1 import ShuffleV1
from .ShuffleNetv2 import ShuffleV2
from .ShuffleNetv2_Imagenet import shufflenet_v2_x0_5
from .ShuffleNetv2_Imagenet import shufflenet_v2_x1_0 as ShuffleNetV2Imagenet
from .ShuffleNetv2_Imagenet import shufflenet_v2_x2_0
from .temp_global import Global_T
from .vgg import vgg8_bn, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from .vggv2 import vgg11_bn as vgg11_imagenet
from .vggv2 import vgg13_bn as vgg13_imagenet
from .wrn import wrn_16_1, wrn_16_2, wrn_40_1, wrn_40_2

cifar100_model_prefix = "/kaggle/input/cifar_teachers/pytorch/default/1/cifar_teachers/"

cifar_model_dict = {
    # teachers
    "resnet56": (
        resnet56,
        cifar100_model_prefix + "resnet56_vanilla/ckpt_epoch_240.pth",
    ),
    "resnet110": (
        resnet110,
        cifar100_model_prefix + "resnet110_vanilla/ckpt_epoch_240.pth",
    ),
    "resnet32x4": (
        resnet32x4,
        cifar100_model_prefix + "resnet32x4_vanilla/ckpt_epoch_240.pth",
    ),
    "ResNet50": (
        ResNet50,
        cifar100_model_prefix + "ResNet50_vanilla/ckpt_epoch_240.pth",
    ),
    "wrn_40_2": (
        wrn_40_2,
        cifar100_model_prefix + "wrn_40_2_vanilla/ckpt_epoch_240.pth",
    ),
    "vgg13": (vgg13_bn, cifar100_model_prefix + "vgg13_vanilla/ckpt_epoch_240.pth"),
    # students
    "resnet8": (resnet8, None),
    "resnet14": (resnet14, None),
    "resnet20": (resnet20, None),
    "resnet32": (resnet32, None),
    "resnet44": (resnet44, None),
    "resnet8x4": (resnet8x4, None),
    "ResNet18": (ResNet18, None),
    "wrn_16_1": (wrn_16_1, None),
    "wrn_16_2": (wrn_16_2, None),
    "wrn_40_1": (wrn_40_1, None),
    "vgg8": (vgg8_bn, None),
    "vgg11": (vgg11_bn, None),
    "vgg16": (vgg16_bn, None),
    "vgg19": (vgg19_bn, None),
    "MobileNetV2": (mobile_half, None),
    "ShuffleV1": (ShuffleV1, None),
    "ShuffleV2": (ShuffleV2, None),
}