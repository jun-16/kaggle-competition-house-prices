import os

import numpy as np
import pandas as pd
import lightgbm as lgb

from model import Model
from util import Util


class ModelLGBM(Model):

    def train(self, tr_x, tr_y, va_x=None, va_y=None):

        # # カテゴリ変数を設定
        # categorical_features = ['OverallQual', 'YearBuilt', 'Neighborhood', 'MSSubClass', 'ExterQual', 'KitchenQual', 'GarageCars', 'BsmtQual', 'FullBath', 'TotRmsAbvGrd', 'FireplaceQu', 'GarageYrBlt', 'haspool', 'has2ndfloor', 'hasgarage', 'hasbsmt', 'hasfireplace']
        # # データセットを生成する
        # lgb_train = lgb.Dataset(tr_x, tr_y, categorical_feature=categorical_features)
        # lgb_eval = lgb.Dataset(va_x, va_y, reference=lgb_train, categorical_feature=categorical_features)

        lgb_train = lgb.Dataset(tr_x, tr_y)
        lgb_eval = lgb.Dataset(va_x, va_y, reference=lgb_train)

        params = dict(self.params)

        validation = va_x is not None
        
        if validation:
            self.model = lgb.train(
                params, lgb_train,
                # モデルの評価用データを渡す
                valid_sets=lgb_eval,
                # 最大で 1000 ラウンドまで学習する
                num_boost_round=1000,
                # 10 ラウンド経過しても性能が向上しないときは学習を打ち切る
                early_stopping_rounds=10
                # ログ
                # callbacks=callbacks
            )
        else:
            self.model = lgb.train(
                params, lgb_train,
                # 最大で 1000 ラウンドまで学習する
                num_boost_round=1000
            )

    def predict(self, te_x):
        return self.model.predict(te_x, num_iteration=self.model.best_iteration)

    def save_model(self):
        model_path = os.path.join('../model/model', f'{self.run_fold_name}.model')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        # best_ntree_limitが消えるのを防ぐため、pickleで保存することとした
        Util.dump(self.model, model_path)

    def load_model(self):
        model_path = os.path.join('../model/model', f'{self.run_fold_name}.model')
        self.model = Util.load(model_path)

    def get_best_score(self):
        return self.model.best_score['valid_0']['rmse']
