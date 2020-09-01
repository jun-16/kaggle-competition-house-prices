import pandas as pd
import numpy as np
import re as re

from base_features import Feature, get_arguments, generate_features

from scipy import stats
from sklearn. preprocessing import LabelEncoder
# from sklearn. preprocessing import PowerTransformer

Feature.dir = '../features'

class TotalSF(Feature):
    def create_features(self):
        train['TotalSF'] = train['1stFlrSF'] + train['2ndFlrSF'] + train['TotalBsmtSF']
        self.train['TotalSF'] = train['TotalSF']
        test['TotalSF'] = test['1stFlrSF'] + test['2ndFlrSF'] + test['TotalBsmtSF']
        self.test['TotalSF'] = test['TotalSF']
        # self.train['TotalSF'], lmbda = stats.boxcox(train['TotalSF'])
        # self.test['TotalSF'], lmbda = stats.boxcox(test['TotalSF'])

class LotArea(Feature):
    def create_features(self):
        self.train['LotArea'], lmbda = stats.boxcox(train['LotArea'])
        self.test['LotArea'], lmbda = stats.boxcox(test['LotArea'])

class GarageYrBlt(Feature):
    def create_features(self):
        self.train['GarageYrBlt'] = train['GarageYrBlt']
        self.test['GarageYrBlt'] = test['GarageYrBlt']

# OverallCond

class GrLivArea(Feature):
    def create_features(self):
        # self.train['GrLivArea'] = train['GrLivArea']
        # self.test['GrLivArea'] = test['GrLivArea']
        self.train['GrLivArea'], lmbda = stats.boxcox(train['GrLivArea'])
        self.test['GrLivArea'], lmbda = stats.boxcox(test['GrLivArea'])

class BsmtFinSF1(Feature):
    def create_features(self):
        self.train['BsmtFinSF1'], lmbda  = stats.yeojohnson(train['BsmtFinSF1'])
        self.test['BsmtFinSF1'], lmbda  = stats.yeojohnson(test['BsmtFinSF1'])

class GarageArea(Feature):
    def create_features(self):
        self.train['GarageArea'] = train['GarageArea']
        self.test['GarageArea'] = test['GarageArea']
        # self.train['GarageArea'], lmbda = stats.boxcox(train['GarageArea'])
        # self.test['GarageArea'], lmbda = stats.boxcox(test['GarageArea'])

#特徴量に1部屋あたりの面積を追加
class FeetPerRoom(Feature):
    def create_features(self):
        train['TotalSF'] = train['1stFlrSF'] + train['2ndFlrSF'] + train['TotalBsmtSF']
        self.train["FeetPerRoom"] =  train["TotalSF"]/train["TotRmsAbvGrd"]

        test['TotalSF'] = test['1stFlrSF'] + test['2ndFlrSF'] + test['TotalBsmtSF']
        self.test["FeetPerRoom"] =  test["TotalSF"]/test["TotRmsAbvGrd"]

class OverallQual(Feature):
    def create_features(self):
        self.train['OverallQual'] = train['OverallQual']
        self.test['OverallQual'] = test['OverallQual']

class OpenPorchSF(Feature):
    def create_features(self):
        # self.train['OpenPorchSF'] = train['OpenPorchSF']
        # self.test['OpenPorchSF'] = test['OpenPorchSF']
        self.train['OpenPorchSF'], lmbda  = stats.yeojohnson(train['OpenPorchSF'])
        self.test['OpenPorchSF'], lmbda  = stats.yeojohnson(test['OpenPorchSF'])

#縁側の合計面積
class Total_porch_sf(Feature):
    def create_features(self):
        self.train['Total_porch_sf'] = (train['OpenPorchSF'] + train['3SsnPorch'] +
                                        train['EnclosedPorch'] + train['ScreenPorch'] +
                                        train['WoodDeckSF'])
        self.test['Total_porch_sf'] = (test['OpenPorchSF'] + test['3SsnPorch'] +
                                        test['EnclosedPorch'] + test['ScreenPorch'] +
                                        test['WoodDeckSF'])

class BsmtUnfSF(Feature):
    def create_features(self):
        # self.train['BsmtUnfSF'] = train['BsmtUnfSF']
        # self.test['BsmtUnfSF'] = test['BsmtUnfSF']
        self.train['BsmtUnfSF'], lmbda  = stats.yeojohnson(train['BsmtUnfSF'])
        self.test['BsmtUnfSF'], lmbda  = stats.yeojohnson(test['BsmtUnfSF'])

class x1stFlrSF(Feature):
    def create_features(self):
        # self.train['x1stFlrSF'] = train['1stFlrSF']
        # self.test['x1stFlrSF'] = test['1stFlrSF']
        self.train['x1stFlrSF'], lmbda = stats.boxcox(train['1stFlrSF'])
        self.test['x1stFlrSF'], lmbda = stats.boxcox(test['1stFlrSF'])

class LotFrontage(Feature):
    def create_features(self):
        # self.train['LotFrontage'] = train['LotFrontage']
        # self.test['LotFrontage'] = test['LotFrontage']
        self.train['LotFrontage'], lmbda  = stats.yeojohnson(train['LotFrontage'])
        self.test['LotFrontage'], lmbda  = stats.yeojohnson(test['LotFrontage'])

#建築した年とリフォームした年の合計
class YearBuiltAndRemod(Feature):
    def create_features(self):
        self.train['YearBuiltAndRemod'] = train['YearBuilt'] + train['YearRemodAdd']
        self.test['YearBuiltAndRemod'] = test['YearBuilt'] + test['YearRemodAdd']

class YearRemodAdd(Feature):
    def create_features(self):
        self.train['YearRemodAdd'] = train['YearRemodAdd']
        self.test['YearRemodAdd'] = test['YearRemodAdd']

class YearBuilt(Feature):
    def create_features(self):
        self.train['YearBuilt'] = train['YearBuilt']
        self.test['YearBuilt'] = test['YearBuilt']

#バスルームの合計面積
class Total_Bathrooms(Feature):
    def create_features(self):
        self.train['Total_Bathrooms'] = (train['FullBath'] + (0.5 * train['HalfBath']) +
                                        train['BsmtFullBath'] + (0.5 * train['BsmtHalfBath']))
        self.test['Total_Bathrooms'] = (test['FullBath'] + (0.5 * test['HalfBath']) +
                                        test['BsmtFullBath'] + (0.5 * test['BsmtHalfBath']))

class WoodDeckSF(Feature):
    def create_features(self):
        # self.train['WoodDeckSF'] = train['WoodDeckSF']
        # self.test['WoodDeckSF'] = test['WoodDeckSF']
        self.train['WoodDeckSF'], lmbda  = stats.yeojohnson(train['WoodDeckSF'])
        self.test['WoodDeckSF'], lmbda  = stats.yeojohnson(test['WoodDeckSF'])

class TotalBsmtSF(Feature):
    def create_features(self):
        # self.train['TotalBsmtSF'] = train['TotalBsmtSF']
        # self.test['TotalBsmtSF'] = test['TotalBsmtSF']
        self.train['TotalBsmtSF'], lmbda = stats.yeojohnson(train['TotalBsmtSF'])
        self.test['TotalBsmtSF'], lmbda = stats.yeojohnson(test['TotalBsmtSF'])

class x2ndFlrSF(Feature):
    def create_features(self):
        self.train['x2ndFlrSF'] = train['2ndFlrSF']
        self.test['x2ndFlrSF'] = test['2ndFlrSF']
        # self.train['x2ndFlrSF'], lmbda = stats.yeojohnson(train['2ndFlrSF'])
        # self.test['x2ndFlrSF'], lmbda = stats.yeojohnson(test['2ndFlrSF'])

class GarageCars(Feature):
    def create_features(self):
        self.train['GarageCars'] = train['GarageCars']
        self.test['GarageCars'] = test['GarageCars']

# Condition1

class Neighborhood(Feature):
    def create_features(self):
        # self.train['Neighborhood'] = train['Neighborhood']
        # self.test['Neighborhood'] = test['Neighborhood']
        le = LabelEncoder()
        le.fit(train['Neighborhood'])
        self.train['Neighborhood'] = le.transform(train['Neighborhood'])

        le.fit(test['Neighborhood'])
        self.test['Neighborhood'] = le.transform(test['Neighborhood'])

# Fireplaces

# SaleCondition

# Bedroom

# Functional

# Exterior1st

# BsmtExposure

class ScreenPorch(Feature):
    def create_features(self):
        self.train['ScreenPorch'] = train['ScreenPorch']
        self.test['ScreenPorch'] = test['ScreenPorch']
        # self.train['ScreenPorch'], lmbda  = stats.yeojohnson(train['ScreenPorch'])
        # self.test['ScreenPorch'], lmbda  = stats.yeojohnson(test['ScreenPorch'])

class MasVnrArea(Feature):
    def create_features(self):
        self.train['MasVnrArea'], lmbda  = stats.yeojohnson(train['MasVnrArea'])
        self.test['MasVnrArea'], lmbda  = stats.yeojohnson(test['MasVnrArea'])

# CentralAirs   

# LotConfig

# YrSold

# MSZoning

class KitchenQual(Feature):
    def create_features(self):
        le = LabelEncoder()
        train['KitchenQual'] = train['KitchenQual'].fillna(train['KitchenQual'].mode()[0])
        le.fit(train['KitchenQual'])
        self.train['KitchenQual'] = le.transform(train['KitchenQual'])

        test['KitchenQual'] = test['KitchenQual'].fillna(test['KitchenQual'].mode()[0])
        le.fit(test['KitchenQual'])
        self.test['KitchenQual'] = le.transform(test['KitchenQual'])

# BsmtFinType1

# HeatingQC

# Foundation

# Kitchen

# LandContour

# MoSold

# HouseStyle

# TotRmsAbvGrd

# GarageType

class MSSubClass(Feature):
    def create_features(self):
        le = LabelEncoder()
        le.fit(train['MSSubClass'])
        self.train['MSSubClass'] = le.transform(train['MSSubClass'])

        le.fit(test['MSSubClass'])
        self.test['MSSubClass'] = le.transform(test['MSSubClass'])

class GarageFinish(Feature):
    def create_features(self):
        le = LabelEncoder()
        train['GarageFinish'] = train['GarageFinish'].fillna('None')
        le.fit(train['GarageFinish'])
        self.train['GarageFinish'] = le.transform(train['GarageFinish'])

        test['GarageFinish'] = test['GarageFinish'].fillna('None')
        le.fit(test['GarageFinish'])
        self.test['GarageFinish'] = le.transform(test['GarageFinish'])

class BsmtQual(Feature):
    def create_features(self):
        le = LabelEncoder()
        train['BsmtQual'] = train['BsmtQual'].fillna(train['BsmtQual'].mode()[0])
        le.fit(train['BsmtQual'])
        self.train['BsmtQual'] = le.transform(train['BsmtQual'])

        test['BsmtQual'] = test['BsmtQual'].fillna(test['BsmtQual'].mode()[0])
        le.fit(test['BsmtQual'])
        self.test['BsmtQual'] = le.transform(test['BsmtQual'])

class EnclosedPorch(Feature):
    def create_features(self):
        # self.train['EnclosedPorch'] = train['EnclosedPorch']
        # self.test['EnclosedPorch'] = test['EnclosedPorch']
        self.train['ScreenPorch'], lmbda  = stats.yeojohnson(train['ScreenPorch'])
        self.test['ScreenPorch'], lmbda  = stats.yeojohnson(test['ScreenPorch'])

class LowQualFinSF(Feature):
    def create_features(self):
        self.train['LowQualFinSF'] = train['LowQualFinSF']
        self.test['LowQualFinSF'] = test['LowQualFinSF']

class BsmtFinSF2(Feature):
    def create_features(self):
        self.train['BsmtFinSF2'], lmbda  = stats.yeojohnson(train['BsmtFinSF2'])
        self.test['BsmtFinSF2'], lmbda  = stats.yeojohnson(test['BsmtFinSF2'])

class MiscVal(Feature):
    def create_features(self):
        # self.train['MiscVal'] = train['MiscVal']
        # self.test['MiscVal'] = test['MiscVal']
        self.train['MiscVal'], lmbda  = stats.yeojohnson(train['MiscVal'])
        self.test['MiscVal'], lmbda  = stats.yeojohnson(test['MiscVal'])

class ExterQual(Feature):
    def create_features(self):
        le = LabelEncoder()
        le.fit(train['ExterQual'])
        self.train['ExterQual'] = le.transform(train['ExterQual'])

        le.fit(test['ExterQual'])
        self.test['ExterQual'] = le.transform(test['ExterQual'])

class FullBath(Feature):
    def create_features(self):
        self.train['FullBath'] = train['FullBath']
        self.test['FullBath'] = test['FullBath']

class TotRmsAbvGrd(Feature):
    def create_features(self):
        self.train['TotRmsAbvGrd'] = train['TotRmsAbvGrd']
        self.test['TotRmsAbvGrd'] = test['TotRmsAbvGrd']

class FireplaceQu(Feature):
    def create_features(self):
        le = LabelEncoder()
        train['FireplaceQu'] = train['FireplaceQu'].fillna('None')
        le.fit(train['FireplaceQu'])
        self.train['FireplaceQu'] = le.transform(train['FireplaceQu'])

        test['FireplaceQu'] = test['FireplaceQu'].fillna('None')
        le.fit(test['FireplaceQu'])
        self.test['FireplaceQu'] = le.transform(test['FireplaceQu'])

#プールの有無
class haspool(Feature):
    def create_features(self):
        self.train['haspool'] = train['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
        self.test['haspool'] = test['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

#2階の有無
class has2ndfloor(Feature):
    def create_features(self):
        self.train['has2ndfloor'] = train['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
        self.test['has2ndfloor'] = test['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

#ガレージの有無
class hasgarage(Feature):
    def create_features(self):
        self.train['hasgarage'] = train['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
        self.test['hasgarage'] = test['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

#地下室の有無
class hasbsmt(Feature):
    def create_features(self):
        self.train['hasbsmt'] = train['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
        self.test['hasbsmt'] = test['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

#暖炉の有無
class hasfireplace(Feature):
    def create_features(self):
        self.train['hasfireplace'] = train['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
        self.test['hasfireplace'] = test['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

# class GarageYrBlt(Feature):
#     def create_features(self):
#         self.train['GarageYrBlt'] = train['GarageYrBlt']
#         self.test['GarageYrBlt'] = test['GarageYrBlt']
#         # カテゴリ変数に変換する
#         train['GarageYrBlt'] = train['GarageYrBlt'].astype(str)
#         test['GarageYrBlt'] = test['GarageYrBlt'].astype(str)

#         le = LabelEncoder()
#         le.fit(train['GarageYrBlt'])
#         self.train['GarageYrBlt'] = le.transform(train['GarageYrBlt'])

#         le.fit(test['GarageYrBlt'])
#         self.test['GarageYrBlt'] = le.transform(test['GarageYrBlt'])

if __name__ == '__main__':
    args = get_arguments()

    train = pd.read_feather('../input/train.feather')
    test = pd.read_feather('../input/test.feather')

    generate_features(globals(), args.force)
