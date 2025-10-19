from functools import partial

from src.modules.batch_norm import *
from src.methods.abstract_meta_learner import AbsMetaLearnerR, AbstractMetaLearner

BATCHNORM = ConventionalBatchNorm
# BATCHNORM = TransductiveBatchNorm

from src.modules.backbones import *
from src.modules import *
from src.methods import *

# Parameters of the model (method and feature extractor)
# Conv4 for cifar&femnist, resnet18 for mini
BACKBONE = ResNet18

H = H_3
# R_4_ADV for femnist, R_RED for others
R = R_RED
TRANSPORTATION_MODULE = OptimalTransport(
    regularization=0.05,
    learn_regularization=False,
    max_iter=1000,
    stopping_criterion=1e-4,
    # power_transform=0.1,
)
R = R_RED
model_list = [ProtoNet, MatchingNet, TransPropNet, TransFineTune]
MODEL = partial(model_list[0])
