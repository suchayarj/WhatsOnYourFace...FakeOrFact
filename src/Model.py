# imports
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Activation, Input, Dropout, LSTM, RepeatVector, Embedding
from tensorflow.keras.models import Sequential

# load data (vectorized using sklearn)
true_samp_vec = pd.read_pickle('data/true_samp_vec_1000.pkl')
decep_samp_vec = pd.read_pickle('data/decep_samp_vec_1000.pkl')

# split data
X_true_train, X_true_test = train_test_split(true_samp_vec.values, test_size = 0.2, random_state = 123)

# MLP model
model = Sequential()
model.add(Dense(25, input_dim=X_true_train.shape[1], activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(X_true_train.shape[1]))
model.compile(loss='mean_squared_error', optimizer='adam')
hist = model.fit(X_true_train,X_true_train,verbose=1,epochs=5)