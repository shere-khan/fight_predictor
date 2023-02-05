import sys, os
from pathlib import Path
# import datetime

import numpy as np
# import tensorflow as tf
# import tensorflow._api.v1.keras.backend as K
import keras
from keras.layers import Dense, Dropout
from keras.regularizers import l2


from utils import r2, random_data_shuffle, get_train_test_data

base_dir = Path("/home/justin/pycharmprojects/fight_predictor/predictor")

def save(model, save_name):
    save_loc = os.path.join(os.getcwd(),
                            'Files', 'Models', save_name)
    model.save(save_loc)

class WinnerModel:
    def __init__(self):
        self.winner_model()
    def winner_model(self):
        """ Model for preidicting overall winner of a bout using static data and predicted bout stats"""

        x_train, y_train, x_test, y_test = get_train_test_data(base_dir, 'Fight_Winner')
        x_train, y_train = random_data_shuffle(x_train, y_train)

        hidden_units = 80
        epochs = 170
        dropout = 0.65
        l2_reg = keras.regularizers.l2(0.001)

        model = keras.models.Sequential()
        model.add(Dense(hidden_units, input_dim=x_train.shape[1], activation='relu',
                        kernel_initializer='normal', kernel_regularizer=l2_reg,
                        name='layer1'))

        model.add(Dropout(dropout))

        model.add(Dense(hidden_units, activation='relu',
                        kernel_initializer='normal', kernel_regularizer=l2_reg,
                        name='layer2'))

        model.add(Dropout(dropout))

        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(0.0001),
                      metrics=['accuracy'])

        model.fit(x_train, y_train, epochs=epochs,
                  batch_size=32, validation_split=0.05)

        scores = model.evaluate(x_test, y_test)
        save(model, 'winner_model.h5')

        print(model.summary())

class FightStatsModel:
    def __init__(self):
        self.fight_stats_model()
    def fight_stats_model(self):
        x_train, y_train, x_test, y_test = get_train_test_data(base_dir, 'Fight_Stats')
        x_train, y_train = random_data_shuffle(x_train, y_train)

        # epochs = 1050
        epochs = 800
        epochs = 1
        hidden1 = 350
        dropout = 0.45
        l2_reg = l2(0.005)

        model = keras.models.Sequential()
        model.add(Dropout(dropout))
        model.add(Dense(hidden1, input_dim=x_train.shape[1], activation='relu',
                        kernel_initializer='normal', kernel_regularizer=l2_reg))

        model.add(Dense(y_train.shape[1], activation='linear'))

        model.compile(
            loss='mse',
            optimizer=keras.optimizers.Adam(0.0001),
            metrics=[r2]
        )

        history = model.fit(x_train, y_train,
                            epochs=epochs, batch_size=32,
                            validation_split=0.1,
                            shuffle=True
                            )

        scores = model.evaluate(x_test, y_test)
        save(model, 'stats_model.h5')

        results = model.predict(x_test)

        predictor_cols = [
            'pass_stat_f1', 'pass_stat_f2', 'str_stat_f1', 'str_stat_f2',
            'sub_stat_f1', 'sub_stat_f2', 'td_stat_f1', 'td_stat_f2'
        ]

        for i in range(0, 20):
            rand_i = np.random.randint(0, 300)
            print('*' * 20)
            for prediction, actual, col in zip(results[rand_i], y_test[rand_i],
                                               predictor_cols):
                print(f'{col}: Prediction= {prediction} Actual = {actual}')

# winner_model()
# fight_stats_model()
