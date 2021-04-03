#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 23:45:29 2020

@author: Mehmood Ali Khan
"""
from sklearn import preprocessing
import pandas as pd
import numpy as np
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import TimeDistributed
from keras.optimizers import RMSprop, SGD, Adam
from keras.initializers import he_normal, he_uniform
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
from pandas import DataFrame
from pandas import concat
import time
from datetime import datetime
#Source : https://machinelearningmastery.com
#/convert-time-series-supervised-learning-problem-python/
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
    
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

class LSTMModel:
    def readViaPandas(self, filename, Header='infer', skipRows=None):
        df = pd.read_csv(filename, header=Header, skiprows=skipRows)
        return df
    def getWindFarmData(self):
        wf1 = self.readViaPandas("WF1DataWithPowerVals.csv")
        wf2 = self.readViaPandas("WF2DataWithPowerVals.csv")
        wf3 = self.readViaPandas("WF3DataWithPowerVals.csv")
        wf4 = self.readViaPandas("WF4DataWithPowerVals.csv")
        wf5 = self.readViaPandas("WF5DataWithPowerVals.csv")
        wf1New = wf1[['u','v','ws','wd','wp']]
        wf2New = wf2[['u','v','ws','wd','wp']]
        wf3New = wf3[['u','v','ws','wd','wp']]
        wf4New = wf4[['u','v','ws','wd','wp']]
        wf5New = wf5[['u','v','ws','wd','wp']]
        
        Wf1 = np.array(wf1New)
        Wf2 = np.array(wf2New)
        Wf3 = np.array(wf3New)
        Wf4 = np.array(wf4New)
        Wf5 = np.array(wf5New)
         
        return Wf1, Wf2, Wf3, Wf4, Wf5
    
    def getCombinedProcessedData(self):
        wf1 = self.readViaPandas("WF1DataWithPowerVals.csv")
        wf2 = self.readViaPandas("WF2DataWithPowerVals.csv")
        wf3 = self.readViaPandas("WF3DataWithPowerVals.csv")
        wf4 = self.readViaPandas("WF4DataWithPowerVals.csv")
        wf5 = self.readViaPandas("WF5DataWithPowerVals.csv")
        
        wf1New = wf1[['u','v','ws','wd','wp']]
        wf2New = wf2[['u','v','ws','wd','wp']]
        wf3New = wf3[['u','v','ws','wd','wp']]
        wf4New = wf4[['u','v','ws','wd','wp']]
        wf5New = wf5[['u','v','ws','wd','wp']]
        
        allWfs = wf1New.append([wf2New,wf3New,wf4New,wf5New])
        allWfsDataSet = np.array(allWfs)
        
        return allWfsDataSet
            
    def preProcessDataForLSTM(self, windFarmDataSet, sequence_length, wfName):
        #min_max_scaler = preprocessing.MinMaxScaler()
        std_scalar     = preprocessing.StandardScaler()
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        result = windFarmDataSet
        result = result[np.all(result != 0.0, axis=1)]
        df = pd.DataFrame({'u': result[:, 0], 'v': result[:, 1], 'ws': result[:, 2], \
                           'wd': result[:, 3], 'wp': result[:, 4] })
        #print df
        df[['u']] = df[['u']].abs()
        df[['v']] = df[['v']].abs()
        df[['u']] = min_max_scaler.fit_transform(df[['u']])
        df[['v']] = min_max_scaler.fit_transform(df[['v']])
        df[['ws']] = min_max_scaler.fit_transform(df[['ws']])
        df[['wd']] = min_max_scaler.fit_transform(df[['wd']])
       
        cols = ['u', 'v', 'ws', 'wd', 'wp']
        df = df[cols]
        df = series_to_supervised(df, 5)
        result = np.array(df)
        print("")
        print wfName+" Dataset: ", result.shape        
        result = result.reshape(result.shape[0], sequence_length, result.shape[1])
        print "Resultant 3D Dataset: ", result.shape
        print('')
        
        #Defining Splitting Ratio to split into Test and Training Datasets
        row = int(round(0.80 * result.shape[0]))
        
        #Splitting into Test and Training Data 20% and 80% respectively.   
        test = result[row:, :, :]
        train = result[:row, :, :]
        #Before Applying minmax Scalar, saving unscaled values
        true_y_test_Unscaled = test[:,:, -1]
        true_y_train_Unscaled = train[:,:, -1]
        
        # reshape input to 2D for Scaling of Data (min_max_scalar)
        train = train.reshape((train.shape[0]*sequence_length, train.shape[2]))
        test = test.reshape((test.shape[0]*sequence_length, test.shape[2]))
        
        #Applying Scaling to Test and Training Data seperately (0,1)
        #train = min_max_scaler.fit_transform(train)
        #test = min_max_scaler.fit_transform(test)
        
        # reshape input to be 3D for LSTM data formats [samples, timesteps, features]
        sequence_length = 1
        train = train.reshape((train.shape[0]/sequence_length, sequence_length, train.shape[1]))
        test = test[:-1]
        test = test.reshape((test.shape[0]/sequence_length, sequence_length, test.shape[1]))
        
        print "Training Dataset: ", train.shape
        print "Testing Dataset:  ", test.shape
        print('')
        
        #Seperating dependant variable i.e wind speed from other variables in testing and training data
        X_train, Y_train = train[:,:, :-1], train[:,:, -1]
        X_test, Y_test = test[:,:, :-1], test[:,:, -1]
        
        
        return X_train, Y_train, X_test, Y_test, test, min_max_scaler
    
    def build_model_LSTMBidirectional(self, X_train):
        model = Sequential()
        layers = [64, 32, 16, 1]
        model.add(Bidirectional(
                LSTM(layers[0], activation='relu',
                    return_sequences=True, kernel_initializer='RandomUniform'), 
                    input_shape=(X_train.shape[1],X_train.shape[2])))
        model.add(Dropout(0.2))
        
        model.add(Bidirectional
                  (LSTM(layers[1], use_bias=True,
                        bias_initializer='zeros',
                        return_sequences=False, activation='relu')))
        model.add(Dropout(0.2))
                
        model.add(Dense(layers[3], activation='relu'))
        
        #print "Fitting Bidirectional LSTM Model"
        print 'Compiling Bidirectional LSTM Model...' 
        
        #Optimizers
        adam    = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-09, decay=0.0001)
        rmsprop = RMSprop(lr=0.001, rho=0.99, epsilon=1e-09, decay=0.0001)
        start   = time.time()
        model.compile(loss='logcosh', optimizer=rmsprop, metrics = ['mse'])
        print "Compilation Completed in ", time.time() - start,"seconds\n"
        return model
    def build_model_LSTMSStacked(self, X_train):
        model = Sequential()
        layers = [64, 32, 16,16,1]
        model.add(LSTM(layers[1], input_shape=(X_train.shape[1], X_train.shape[2]),
        return_sequences=True))
                   
        model.add(LSTM( layers[2],activation='relu',
        return_sequences=True))
    
        model.add(LSTM( layers[3], 
        return_sequences=False))
                      
        model.add(Dense(layers[4], activation='relu'))
        
        print 'Compiling Stacked LSTM Model...' 
        
        #Optimizers
        adam    = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-09, decay=0.0001)
        rmsprop = RMSprop(lr=0.001, rho=0.99, epsilon=1e-09, decay=0.0001)
        start   = time.time()
        model.compile(loss='logcosh', optimizer=rmsprop, metrics = ['mse'])
        print "Compilation Completed in ", time.time() - start,"seconds\n"
        return model
    
    def modelLSTMfit(self, modelLSTM, X_train, Y_train,X_test, Y_test, nepochs, WindFarm ):
        try:
            print "Fitting The model to Data\n"
                  
            modelHistory = modelLSTM.fit(X_train, Y_train,
                                batch_size=32,
                                epochs=nepochs,
                                validation_split=0.2,
                                shuffle=True)
            
            print "Model Fitting Completed...!"
            #print 'Training duration (s) : ', time.time() - global_start_time
        except KeyboardInterrupt:
            print('')
            print ("Some Error or Intruppt Occured")
        predicted = modelLSTM.predict(X_test)
        Y__test_predicted = predicted.reshape((predicted.shape[0]))
        
    
        Y_test = Y_test.reshape((Y_test.shape[0]))
        
        rmse = sqrt(mean_squared_error(Y_test, Y__test_predicted))
        mae  = mean_absolute_error(Y_test, Y__test_predicted)
        stde = np.std(Y__test_predicted)
    
        
        print("MAE:", mae)
        print("STDE:", stde)
        print("RMSE:", rmse)
 
        return Y_test[:170], Y__test_predicted[:170]
    
    def plotGraph(self, actual, predicted, windfarm, modeltype):
       
        plt.plot(actual, color='#B22400')
        plt.plot(predicted, color='#006BB2', ls='dashed')
        plt.xlabel('Time(Hours)')
        plt.ylabel('Wind Power Prediction')
        plt.xlim(1,170)
        plt.title(windfarm+': '+modeltype)
        plt.legend(['Actual', 'Predicted'])
        plt.show()
        rmse = sqrt(mean_squared_error(actual, predicted))
        mae  = mean_absolute_error(actual, predicted)
        stde = np.std(predicted)
       
        print("MAE:", mae)
        print("STDE:", stde)
        print("RMSE:", rmse)
        
