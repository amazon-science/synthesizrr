from typing import *
import os
import numpy as np
import pandas as pd

from synthergent.base.util import optional_dependency
from synthergent.base.data import FileMetadata
from synthergent.base.framework import ClassificationData, Classifier, EncodingRange
from synthergent.base.constants import Storage, MLType

with optional_dependency('sklearn', 'joblib', error='raise'):
    import joblib
    from sklearn.linear_model import SGDClassifier


    class SklearnSGDClassifier(Classifier):
        label_encoding_range = EncodingRange.ZERO_TO_N_MINUS_ONE

        model: SGDClassifier = None

        class Hyperparameters(Classifier.Hyperparameters):
            alpha: float = 3e-1  ## Custom default learning-rate
            loss: str = 'log_loss'

        def initialize(self, model_dir: Optional[FileMetadata] = None):
            if model_dir is None:
                self.model: SGDClassifier = SGDClassifier(
                    **self.hyperparams.dict(include=SGDClassifier.__init__),
                    random_state=self.hyperparams.seed,
                )
            else:
                assert model_dir.storage is Storage.LOCAL_FILE_SYSTEM, 'Can only load models from disk.'
                self.model: SGDClassifier = joblib.load(os.path.join(model_dir.path, 'model.pkl'))

        def train_step(self, batch: ClassificationData, **kwargs):
            ## Convert from our internal format to Pandas Series
            features: pd.DataFrame = batch.features(MLType.FLOAT, MLType.INT).pandas()
            ground_truths: pd.Series = batch.ground_truths().pandas()  ## 0, ..., N-1
            if getattr(self.model, "classes_", None) is None:
                self.model.partial_fit(features, ground_truths, classes=np.array(self.encoded_labelspace))
            else:
                self.model.partial_fit(features, ground_truths)

        def predict_step(self, batch: ClassificationData, **kwargs) -> Dict:
            ## Convert from our internal format to Pandas DataFrame:
            features: pd.DataFrame = batch.features(MLType.FLOAT, MLType.INT).pandas()
            scores: np.ndarray = self.model.predict_proba(features)
            if np.isnan(scores).all():  ## https://github.com/scikit-learn/scikit-learn/issues/17978
                scores: np.ndarray = np.full_like(scores, fill_value=1.0, dtype=np.float) / self.num_labels
            return {'scores': scores, 'labels': self.model.classes_}

        def save(self, model_dir: FileMetadata):
            joblib.dump(self.model, os.path.join(model_dir.path, 'model.pkl'))
