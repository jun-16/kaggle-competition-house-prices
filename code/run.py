import numpy as np
import pandas as pd

from model_nn import ModelNN
from model_lgbm import ModelLGBM
from runner import Runner
from util import Submission

if __name__ == '__main__':

    params_lgbm = {
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "max_depth": 4,
        "colsample_bytree": 0.9,
        "subsample": 0.9,
        "reg_alpha": 0.1,
        "reg_lambda": 0.0,
        "min_child_weight": 1,
        "num_leaves": 31
    }

    params_lgbm_all = dict(params_lgbm)
    params_lgbm_all['num_round'] = 350

    params_nn = {
        'layers': 3,
        'nb_epoch': 10000,
        'patience': 100,
        'dropout': 0.5,
        'units': 100,
    }

    # 特徴量の指定
    features = [
        "TotalSF",
        "x1stFlrSF",
        "x2ndFlrSF",
        "TotalBsmtSF",
        "GrLivArea",
        "LotArea",
        "GarageArea",
        "YearRemodAdd",
        "BsmtFinSF1",
        "MasVnrArea",
        "LotFrontage",
        "LowQualFinSF",
        "ScreenPorch",
        "EnclosedPorch",
        "BsmtFinSF2",
        "WoodDeckSF",
        "BsmtUnfSF",
        "MiscVal",
        "OpenPorchSF",
        "FeetPerRoom",
        "YearBuiltAndRemod",
        "Total_Bathrooms",
        "Total_porch_sf",
        "OverallQual",
        "YearBuilt",
        "Neighborhood",
        "MSSubClass",
        "ExterQual",
        "KitchenQual",
        "GarageCars",
        "BsmtQual",
        "FullBath",
        "TotRmsAbvGrd",
        "FireplaceQu",
        "GarageYrBlt",
        "haspool",
        "has2ndfloor",
        "hasgarage",
        "hasbsmt",
        "hasfireplace"
    ]

    # lightgbmによる学習・予測
    runner = Runner('lgbm1', ModelLGBM, features, params_lgbm)
    runner.run_train_cv()
    runner.run_predict_cv()
    Submission.create_submission('lgbm1')

    # ニューラルネットによる学習・予測
    # runner = Runner('nn1', ModelNN, features, params_nn)
    # runner.run_train_cv()
    # runner.run_predict_cv()
    # Submission.create_submission('nn1')

    # # (参考）lightgbmによる学習・予測 - 学習データ全体を使う場合
    # runner = Runner('lgbm1-train-all', ModelLGBM, features, params_lgbm_all)
    # runner.run_train_all()
    # runner.run_predict_all()
    # Submission.create_submission('lgbm1-train-all')
