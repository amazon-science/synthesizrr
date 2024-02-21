from synthesizrr.base.util import AutoEnum, auto, alias

K_FOLD_NAME_PREFIX = 'fold_'


class DataSplit(AutoEnum):
    TRAIN = alias('TRAINING')  ## Used during training pipeline, has ground-truths.
    VALIDATION = alias('EVAL', 'VAL', 'VALID', 'DEV')  ## Used during training pipeline, has ground-truths.
    TRAIN_VAL = alias('TRAIN_VALIDATION')  ## Used during training pipeline, has ground-truths.
    TEST = auto()  ## Used during training pipeline, has ground-truths.
    UNSUPERVISED = auto()  ## Used during training pipeline, expected not to have ground-truths.
    PREDICT = auto()  ## Used during real-world inference, expected not to have ground-truths.


DatasetSplit = DataSplit


class HpoStrategy(AutoEnum):
    Random = auto()
    Bayesian = auto()


class InferenceMode(AutoEnum):
    SAGEMAKER = auto()
    SAGEMAKER_END_TO_END = auto()
    CPP_END_TO_END = auto()
