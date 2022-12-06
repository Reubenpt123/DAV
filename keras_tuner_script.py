import pandas as pd
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
import time
import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from math import sqrt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import *
from keras.utils import np_utils
LOG_DIR = f'../{int(time.time())}'

df = pd.read_csv('d:\My Drive\Colab Notebooks\DAV Assessment\cefas_smartBuoy\clean_ext_data.csv', parse_dates = ['dateTime'])
df = df.set_index('dateTime')
df = df.drop('kd', axis = 1)
df = df.dropna()

phase_0 = df.loc[df['phase'] == 0, :].copy()
X = phase_0.drop(['fluors', 'phase'], axis = 1).copy()
y = phase_0['fluors'].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X = scaler.transform(X)

def build_model(hp):  # random search passes this hyperparameter() object 
    model = Sequential()
    model.add(Dense(hp.Int('input_units', min_value = 50, max_value = 1000, step = 50), activation = "relu", input_shape = (4, )))
    for i in range(hp.Int('n_layers', 1, 5)):
        model.add(Dense(units = hp.Int(f'dense_layer_{i}_units', min_value = 50, max_value = 1000, step = 50), activation = "relu"))
        model.add(Dropout(hp.Float(f'dropout_layer_{i}_%', min_value = 0.0, max_value = 0.4, step = 0.1)))
    model.summary()
    model.compile(loss = "mean_squared_error", optimizer = "adam", metrics = ['mean_squared_error'])
    return model

tuner = RandomSearch(
    build_model,
    objective = 'val_mean_squared_error',
    max_trials = 5,  # how many model variations to test?
    executions_per_trial = 1,  # how many trials per variation? (same model could perform differently)
    directory = LOG_DIR)

tuner.search(x = X_train,
             y = y_train,
             verbose = 1,
             epochs = 2,
             batch_size = 64,
             validation_data = (X_test, y_test))

tuner.get_best_hyperparameters()[0].values
tuner.get_best_models()[0].summary()