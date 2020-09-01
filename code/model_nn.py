import os

import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.layers.advanced_activations import PReLU
from keras.layers.core import Activation, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, load_model
from keras.utils import np_utils
from keras.optimizers import Adam, Adadelta
from sklearn.preprocessing import StandardScaler

from model import Model
from util import Util

import keras.backend as K

# tensorflowの警告抑制
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# 再現性のある乱数を生成
np.random.seed(seed=0)

# os.environ['PYTHONHASHSEED'] = '0'
# random.seed(0)

session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1
)

tf.set_random_seed(0)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

class ModelNN(Model):

    def train(self, tr_x, tr_y, va_x=None, va_y=None):

        # データのセット・スケーリング
        validation = va_x is not None
        scaler = StandardScaler()
        scaler.fit(tr_x)
        tr_x = scaler.transform(tr_x)

        if validation:
            va_x = scaler.transform(va_x)

        # パラメータ
        layers = self.params['layers']
        dropout = self.params['dropout']
        units = self.params['units']
        nb_epoch = self.params['nb_epoch']
        patience = self.params['patience']

        # モデルの構築
        model = Sequential()
        model.add(Dense(units, input_shape=(tr_x.shape[1],), activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))

        for l in range(layers - 1):
            model.add(Dense(units, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))

        # # モデルの構築
        # model = Sequential()
        # model.add(Dense(512, input_shape=(tr_x.shape[1],), activation='relu'))
        # model.add(BatchNormalization())
        # model.add(Dropout(dropout))

        # model.add(Dense(256, activation='relu'))
        # model.add(BatchNormalization())
        # model.add(Dropout(dropout))

        # model.add(Dense(128, activation='relu'))
        # model.add(BatchNormalization())
        # model.add(Dropout(dropout))

        # model.add(Dense(64, activation='relu'))
        # model.add(BatchNormalization())
        # model.add(Dropout(dropout))
        
        model.add(Dense(1))
        optimizer = Adam(lr=0.005, decay=0.)
        # optimizer = Adadelta()
        model.compile(loss='mean_squared_error', optimizer=optimizer)

        if validation:
            early_stopping = EarlyStopping(monitor='val_loss', patience=patience,
                                           verbose=1, restore_best_weights=True)
            model.fit(tr_x, tr_y, epochs=nb_epoch, batch_size=128, verbose=2,
                      validation_data=(va_x, va_y), callbacks=[early_stopping])
            # model.fit(tr_x, tr_y, epochs=nb_epoch, batch_size=128, verbose=2,
            #           validation_data=(va_x, va_y))
        else:
            model.fit(tr_x, tr_y, nb_epoch=nb_epoch, batch_size=128, verbose=2)

        # モデル・スケーラーの保持
        self.model = model
        self.scaler = scaler

    def predict(self, te_x):
        te_x = self.scaler.transform(te_x)
        pred = self.model.predict_proba(te_x)
        return pred

    def save_model(self):
        model_path = os.path.join('../model/model', f'{self.run_fold_name}.h5')
        scaler_path = os.path.join('../model/model', f'{self.run_fold_name}-scaler.pkl')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)
        Util.dump(self.scaler, scaler_path)

    def load_model(self):
        model_path = os.path.join('../model/model', f'{self.run_fold_name}.h5')
        scaler_path = os.path.join('../model/model', f'{self.run_fold_name}-scaler.pkl')
        self.model = load_model(model_path)
        self.scaler = Util.load(scaler_path)
