from typing import *
import os
import numpy as np
import pandas as pd

from synthesizrr.base.util import optional_dependency
from synthesizrr.base.data import FileMetadata
from synthesizrr.base.framework import Regressor, RegressionData
from synthesizrr.base.constants import Storage, MLType

with optional_dependency('sklearn', 'joblib'):
    import joblib
    from sklearn.linear_model import SGDRegressor


    class SklearnSGDRegressor(Regressor):
        model: SGDRegressor = None

        class Hyperparameters(Regressor.Hyperparameters):
            alpha: float = 1e-1  ## Custom default learning-rate

        def initialize(self, model_dir: Optional[FileMetadata] = None):
            if model_dir is None:
                self.model: SGDRegressor = SGDRegressor(
                    **self.hyperparams.dict(include=SGDRegressor.__init__),
                    random_state=self.hyperparams.seed,
                )
            else:
                assert model_dir.storage is Storage.LOCAL_FILE_SYSTEM, 'Can only load models from disk.'
                self.model: SGDRegressor = joblib.load(os.path.join(model_dir.path, 'model.pkl'))

        def train_step(self, batch: RegressionData, **kwargs):
            ## Convert from our internal format to Pandas Series
            features: pd.DataFrame = batch.features(MLType.FLOAT, MLType.INT).pandas()
            ground_truths: pd.DataFrame = batch.ground_truths().pandas()
            self.model.partial_fit(features, ground_truths)

        def predict_step(self, batch: RegressionData, **kwargs) -> np.ndarray:
            ## Convert from our internal format to Pandas DataFrame:
            features: pd.DataFrame = batch.features(MLType.FLOAT, MLType.INT).pandas()
            predictions = self.model.predict(features)
            return predictions

        def save(self, model_dir: FileMetadata):
            joblib.dump(self.model, os.path.join(model_dir.path, 'model.pkl'))
