# imports
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Input, Dropout, LSTM, RepeatVector, Embedding
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# load data (vectorized using sklearn)
true_samp_vec = pd.read_pickle('data/true_samp_vec_1000.pkl')
decep_samp_vec = pd.read_pickle('data/decep_samp_vec_1000.pkl')

# split data
X_true_train, X_true_test = train_test_split(true_samp_vec.values, test_size = 0.2, random_state = 123)

# MLP model (using Review & Rating features)
model = Sequential()
model.add(Dense(25, input_dim=X_true_train.shape[1], activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(X_true_train.shape[1]))
model.compile(loss='mean_squared_error', optimizer='adam')
hist = model.fit(X_true_train,X_true_train,verbose=1,epochs=5)

pred = model.predict(X_true_test)
score1 = np.sqrt(metrics.mean_squared_error(pred,X_true_test))
pred = model.predict(true_samp_vec.values)
score2 = np.sqrt(metrics.mean_squared_error(pred,true_samp_vec.values))
pred = model.predict(decep_samp_vec.values)
score3 = np.sqrt(metrics.mean_squared_error(pred,decep_samp_vec.values))
print(f"Insample True Reviews Score (RMSE): {score1}".format(score1))
print(f"Out of Sample True Reviews Score (RMSE): {score2}")
print(f"Deceptive Reviews Score (RMSE): {score3}")



# LSTM Model (using Review & Rating features)
# Load data 
true = pd.read_csv('true.csv').drop(columns=['Unnamed: 0', 'True(1)/Deceptive(0)'])
decep = pd.read_csv('decep.csv').drop(columns=['Unnamed: 0', 'True(1)/Deceptive(0)'])
true_samp = true.sample(100000, random_state = 123)
decep_samp = decep.sample(100000, random_state = 123)

#tokenize 
tokenizer = Tokenizer(num_words=5000, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
#true_samp
tokenizer.fit_on_texts(true_samp.Review.values)
word_index = tokenizer.word_index
X = tokenizer.texts_to_sequences(true_samp.Review.values)
X = pad_sequences(X, maxlen=300)

#decep_samp (to test the model later)
tokenizer.fit_on_texts(decep_samp[:80000].Review.values)
word_index_decep = tokenizer.word_index
X_decep = tokenizer.texts_to_sequences(decep_samp[:80000].Review.values)
X_decep = pad_sequences(X_decep, maxlen=300)

#add rating (star) feature to the array
stars =true_samp.Stars.values
X = np.hstack((X, stars.reshape(100000,1)))

decep_stars =decep_samp.Stars.values
X_decep = np.hstack((X_decep, decep_stars.reshape(100000,1)))

#split data for test and train
X_train, X_test = train_test_split(X, test_size = 0.2, random_state=123)

#reshape for LSTM model
X_train2 = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test2 = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

def autoencoder_model_new(X):
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    L1 = LSTM(40, activation = 'relu', return_sequences = True)(inputs)
    L2 = LSTM(5, activation = 'relu', return_sequences = False)(L1)
    L3 = RepeatVector(X.shape[1])(L2)
    L4 = LSTM(5, activation = 'relu', return_sequences = True)(L3)
    L5 = LSTM(40, activation= 'relu', return_sequences = True)(L4)
    output = Dense(X.shape[2])(L5)
    model = Model(inputs=inputs, outputs=output)
    return model

model = autoencoder_model_new(X_train2)
model.compile(optimizer='adam', loss = 'mae')
model.summary()
history = model.fit(X_train2, X_train2, epochs=10, batch_size = 10, validation_split = 0.1).history

#plot training & validation losses
fig,ax = plt.subplots(figsize=(14,6), dpi=80)
ax.plot(history['loss'],'b', label='Train', linewidth=2)
ax.plot(history['val_loss'], 'r', label ='Validation', linewidth = 2)
ax.set_title('Model Loss', fontsize=16)
ax.set_ylabel('Loss (rmse)')
ax.set_xlabel('Epoch')
ax.legend(loc='upper right')
plt.show()

#plot distribution of training losses
X_pred = model.predict(X_train2)
X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
X_pred = pd.DataFrame(X_pred, columns = np.arange(X_train2.shape[2]))
X_pred.index = np.arange(X_train2.shape[0])

scored = pd.DataFrame(index = np.arange(X_train2.shape[0]))
Xtrain = X_train2.reshape(X_train2.shape[0], X_train2.shape[2])
scored['Loss_mae'] = np.mean(np.abs(X_pred-Xtrain), axis = 1)
plt.figure(figsize=(16,9), dpi=80)
plt.title('Loss Distribution', fontsize= 16)
sns.distplot(scored['Loss_mae'], bins =20, kde= True, color= 'blue')

#plot X_decep loss distribution
X_pred = model.predict(X_decep2)
X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
X_pred = pd.DataFrame(X_pred, columns = np.arange(X_decep2.shape[2]))
X_pred.index = np.arange(X_decep2.shape[0])

scored = pd.DataFrame(index = np.arange(X_decep2.shape[0]))
Xdecep = X_decep2.reshape(X_decep2.shape[0], X_decep2.shape[2])
scored['Loss_mae'] = np.mean(np.abs(X_pred-Xdecep), axis = 1)
plt.figure(figsize=(16,9), dpi=80)
plt.title('Loss Distribution', fontsize= 16)
sns.distplot(scored['Loss_mae'], bins =20, kde= True, color= 'blue')