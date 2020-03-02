# imports
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
plt.style.use('fivethirtyeight')
from sklearn import metrics
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Activation, Input, Dropout, LSTM, RepeatVector, Embedding
from tensorflow.keras.models import Sequential



# MLP autoencoder model with 3 layers
def autoencoder(vectorized_data, unit_list = [25,3], epoch = 5):
    '''
    input
    vectorized_data: DataFrame/ dataframe that's been cleaned and vectorized using functions in CleanText.py
    unit_list: list/ list of 2 positive integers, dimensionality of the output space

    output
    model
    '''
    model = Sequential()
    model.add(Dense(units = unit_list[0], input_dim=vectorized_data.shape[1], activation='relu'))
    model.add(Dense(units = unit_list[1], activation='relu'))
    model.add(Dense(units = unit_list[0], activation='relu'))
    model.add(Dense(vectorized_data.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    hist = model.fit(vectorized_data,vectorized_data,verbose=1,epochs=epoch)
    return model

def generate_loss_list(model, vectorized_data):
    loss_list = []
    for i in range(cleaned_data.shape[0]):
        loss_list.append(model.evaluate(vectorized_data[i].reshape(1,1001),vectorized_data[i].reshape(1,1001))[0])
    return loss_list

def plot_loss(loss_list, figname):
    '''
    input 
    true_loss: list/ list of loss values from true Yelp reviews
    decep_loss: list/ list of loss values from deceptive Yelp reviews
    
    output
    distribution plot
    '''
    plt.figure(figsize=(12,6), dpi=80)
    plt.title('Loss Distribution (Yelp)', fontsize= 16)
    sns.distplot(loss_list, bins =30, kde= True, label=True)
    plt.xlabel('Loss (MSE)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('img/{}'.format(figname))