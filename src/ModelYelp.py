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

# load pre-processed Yelp data (vectorized using sklearn)
true_samp_vec = pd.read_pickle('data/true_samp_vec_1000.pkl')
decep_samp_vec = pd.read_pickle('data/decep_samp_vec_1000.pkl')

# split data
def train_test_split(true_samp_vec):
    '''
    input
    true_samp_vec: DataFrame/ vectorized Yelp true review dataset 
    '''
    X_true_train, X_true_test = train_test_split(true_samp_vec.values, test_size = 0.2, random_state = 123)
    return X_true_train, X_true_test

# MLP autoencoder model with 3 layers
def autoencoder(X_true_train, unit_list = [25,3], epoch = 5):
    '''
    input
    unit_list: list/ list of 2 positive integers, dimensionality of the output space

    output
    model
    '''
    model = Sequential()
    model.add(Dense(units = unit_list[0], input_dim=X_true_train.shape[1], activation='relu'))
    model.add(Dense(units = unit_list[1], activation='relu'))
    model.add(Dense(units = unit_list[0], activation='relu'))
    model.add(Dense(X_true_train.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    hist = model.fit(X_true_train,X_true_train,verbose=1,epochs=epoch)
    return model

# evaluate model and get a list of loss values
def generate_loss_list(model, X_true_train, decep_samp_vec):
    true_loss = []
    decep_loss = []
    for i in range(X_true_train.shape[0]):
        X_true_train_loss.append(model.evaluate(X_true_train[i].reshape(1,1001),X_true_train[i].reshape(1,1001))[0])
        decep_loss.append(model3.evaluate(decep_samp_vec.values[i].reshape(1,1001),decep_samp_vec.values[i].reshape(1,1001))[0])
    return true_loss, decep_loss

#plot distribution to determine anamoly threshold 
def plot_loss(true_loss, decep_loss, figname):
    '''
    input 
    true_loss: list/ list of loss values from true Yelp reviews
    decep_loss: list/ list of loss values from deceptive Yelp reviews
    
    output
    distribution plot
    '''
    plt.figure(figsize=(12,6), dpi=80)
    plt.title('Loss Distribution (Yelp)', fontsize= 16)
    sns.distplot(decep_loss, bins =30, kde= True, color= 'red', label=True)
    sns.distplot(true_loss, bins =30, kde= True, color= 'blue', label=True)
    plt.legend(['Deceptive Reviews', 'Authentic Reviews'], fontsize = 'large')
    plt.xlabel('Loss (MSE)')
    plt.ylabel('Count')
    plt.xlim([0.0007,0.0012])
    plt.tight_layout()
    plt.savefig('img/{}'.format(figname))


def calculate_confusion(true_frame,decep_frame, threshold):
    '''
    input
    true_frame: DataFrame/ loss(MSE) dataframe for true reviews
    decep_frame: DataFrame/ loss(MSE) dataframe for deceptive reviews
    threshold: float/ threshold to determine anomaly (if loss> threshold, then it anomaly)
    
    output
    FP, TN, TP, FN
    confusion matrix
    '''
    true_frame['Threshold'] = threshold
    true_frame['Anomaly'] = true_frame['Loss(MSE)'] > true_frame['Threshold']
       
    decep_frame['Threshold'] = threshold
    decep_df['Anomaly'] = decep_frame['Loss(MSE)'] > decep_frame['Threshold']

    FP, TN, TP, FN = true_frame['Anomaly'].values.sum(), (~true_frame['Anomaly']).values.sum(), decep_frame['Anomaly'].values.sum(), (~decep_frame['Anomaly']).values.sum()
    
    
    cm = np.array([[TN, FP],[FN, TP]])
    fig, ax= plt.subplots(figsize = (5,5))
    sns.heatmap(cm, ax=ax, annot=True, fmt='d')
    ax.set_xlabel('Predicted labels', fontsize = 15);ax.set_ylabel('True labels', fontsize = 15); 
    ax.set_title('Confusion Matrix (Threshold = {})'.format(threshold), fontsize =15)
    plt.tight_layout()
    plt.savefig('img/CM{}.png'.format(threshold))
        
    return FP, TN, TP, FN

def precision_recall(true_frame,decep_frame, threshold):
    '''
    input
    true_frame: DataFrame/ loss(MSE) dataframe for true reviews
    decep_frame: DataFrame/ loss(MSE) dataframe for deceptive reviews
    threshold: float/ threshold to determine anomaly (if loss> threshold, then it anomaly)
    
    output
    FP, TN, TP, FN, precision, recall
    '''
    true_frame['Threshold'] = threshold
    true_frame['Anomaly'] = true_frame['Loss(MSE)'] > true_frame['Threshold']
       
    decep_frame['Threshold'] = threshold
    decep_frame['Anomaly'] = decep_frame['Loss(MSE)'] > decep_frame['Threshold']

    FP, TN, TP, FN = true_frame['Anomaly'].values.sum(), (~true_frame['Anomaly']).values.sum(), decep_frame['Anomaly'].values.sum(), (~decep_frame['Anomaly']).values.sum()
    
    #calculate precision and recall
    precision = (TP)/(TP+FP)
    recall = (TP)/(TP+FN)
        
    return precision, recall   

#generate precision and recall lists from different thresholds
def precision_recall_list(true_frame, decep_frame, threshold_range):
    precisions = []
    recalls = []
    thresholds = np.arange(min(decep_loss), max(decep_loss),0.000001)
    for i in thresholds:
        p, r = precision_recall(true_frame, decep_frame, i)
        precisions.append(p)
        recalls.append(r)
    return precisions, recalls

#plot precision-recall curve
def plot_precision_recall(precisions, recalls):
    '''
    input
    precisions: list of precisions values
    recalls: list of precisions values

    output
    precicion-recall curve
    '''
    fig, ax = plt.subplots(figsize = (7,4))
    plt.plot(recalls, precisions, color = 'red',)
    plt.title('Precision-Recall Curve', fontsize = 20 )
    plt.xlabel('Recall', fontsize=15)
    plt.ylabel('Precision', fontsize =15)
    plt.xlim([0.01,1.02])
    plt.grid(b=None)
    ax.axhline(0.5, color = 'black', linestyle='--')
    plt.tight_layout()
