
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 00:42:56 2019

@author: Mehmood Ali Khan
"""

import time
from DataPreprocessing import DataPreprocessing
from DataReader import DataReader
from LSTMModel import LSTMModel

if __name__=='__main__':    
    objReader = DataReader()
    objDataPrep = DataPreprocessing()
    #############################################################
    ## Data preprocessing, combining date and hour offset to creat date time column,
    ## combining observed wind power values with atomospheric data of wind farms
    ## uncomment the below code to check data preprocessing steps
    """
    print("\nStarted reading wind farms Data csv files along with train.csv file....\n")
    train, wf1, wf2, wf3, wf4, wf5 = objReader.readData()
    print("Reading of All Wind Farms data files along with train.csv file completed..!\n")
    
    print("Started data preparation of individual Wind Farms wf1, wf2, . . . wf5 ...\n")
    objDataPrep = DataPreprocessing()
    Wf1WithDatetime = objDataPrep.prepareWindFarmDateTimeColumn(wf1, 1)
    print("Wind Farm--1 DataTime Column calculated using date & hour columns...\n")
    
    Wf2WithDatetime = objDataPrep.prepareWindFarmDateTimeColumn(wf2, 2)
    print("Wind Farm--2 DataTime Column calculated using date & hour columns...\n")

    Wf3WithDatetime = objDataPrep.prepareWindFarmDateTimeColumn(wf3, 3)
    print("Wind Farm--3 DataTime Column calculated using date & hour columns...\n")
    
    Wf4WithDatetime = objDataPrep.prepareWindFarmDateTimeColumn(wf4, 4)
    print("Wind Farm--4 DataTime Column calculated using date & hour columns...\n")
     
    Wf5WithDatetime = objDataPrep.prepareWindFarmDateTimeColumn(wf5, 5)
    print("Wind Farm--5 DataTime Column calculated using date & hour columns...\n")
    
    del wf1, wf2, wf3, wf4, wf5
    
    print('Reading Training Data file\n')
    train = objReader.readTrainingData("train.csv")
    
    print('Reading Wind Farm-01 Data')
    wf1WithDateTime = objReader.readWindFarm("wf1_dateTime_col.csv")
    
    print('Reading Wind Farm-02 Data')
    wf2WithDateTime = objReader.readWindFarm("wf2_dateTime_col.csv")

    print('Reading Wind Farm-03 Data')
    wf3WithDateTime = objReader.readWindFarm("wf3_dateTime_col.csv")

    print('Reading Wind Farm-04 Data')
    wf4WithDateTime = objReader.readWindFarm("wf4_dateTime_col.csv")

    print('Reading Wind Farm-05 Data')
    wf5WithDateTime = objReader.readWindFarm("wf5_dateTime_col.csv")
    
    print("Preparing Wind Farm-01 Data with Power Values from training file Data")
    wf1WithPowerVals = objDataPrep.combinePowerWithForecastVals(train, wf1WithDateTime, 1)
    np.savetxt("WF1DataWithPowerVals.csv",wf1WithPowerVals,fmt='%s,%s,%s,%s,%s,%s')
    
    print("Preparing Wind Farm-02 Data with Power Values from training file Data")
    wf2WithPowerVals = objDataPrep.combinePowerWithForecastVals(train, wf2WithDateTime, 2)
    np.savetxt("WF2DataWithPowerVals.csv",wf2WithPowerVals,fmt='%s,%s,%s,%s,%s,%s')
    
    print("Preparing Wind Farm-03 Data with Power Values from training file Data")
    wf3WithPowerVals = objDataPrep.combinePowerWithForecastVals(train, wf3WithDateTime, 3)
    np.savetxt("WF3DataWithPowerVals.csv",wf3WithPowerVals,fmt='%s,%s,%s,%s,%s,%s')
    
    print("Preparing Wind Farm-04 Data with Power Values from training file Data")
    wf4WithPowerVals = objDataPrep.combinePowerWithForecastVals(train, wf4WithDateTime, 4)
    np.savetxt("WF4DataWithPowerVals.csv",wf4WithPowerVals,fmt='%s,%s,%s,%s,%s,%s')
    
    print("Preparing Wind Farm-05 Data with Power Values from training file Data")
    wf5WithPowerVals = objDataPrep.combinePowerWithForecastVals(train, wf5WithDateTime, 5)
    np.savetxt("WF5DataWithPowerVals.csv",wf5WithPowerVals,fmt='%s,%s,%s,%s,%s,%s')
    """
    
    objLSTM = LSTMModel()
    
    Windfarm1, Windfarm2, Windfarm3, Windfarm4, Windfarm5 = \
    objLSTM.getWindFarmData()
    nepochs = 5
   
    
    #Wind Farm 1 training and prediction
    X_train, Y_train, X_test, Y_test, test, min_max_scaler = \
    objLSTM.preProcessDataForLSTM(Windfarm1, 1, 'Wind Farm-1')
    
    modelLSTMbi = objLSTM.build_model_LSTMBidirectional(X_train)
    #modeltype = 'Bidirectional LSTM'
    modelLSTMstacked = objLSTM.build_model_LSTMSStacked(X_train)
    #modelLSTMstacked
    #modeltype = 'Stacked LSTM'
    global_start_time = time.time()
        
    #Results= open("Testresults.txt","a+")
    #Results.write("\n"+modeltype+"\r\n")
    #Results.close
    
    actualwf1, predictedwf1 = objLSTM.modelLSTMfit(modelLSTMbi, X_train, Y_train,\
                                                   X_test, Y_test, nepochs, 'Wind Farm-1')
    objLSTM.plotGraph(actualwf1, predictedwf1, 'Wind Farm-1', 'Bidirectional LSTM')
    actualwf1, predictedwf1 = objLSTM.modelLSTMfit(modelLSTMstacked, X_train, Y_train,\
                                                   X_test, Y_test, nepochs, 'Wind Farm-1')
    objLSTM.plotGraph(actualwf1, predictedwf1, 'Wind Farm-1', 'Stacked LSTM')
    
    #Wind Farm 2 training and prediction
    X_train, Y_train, X_test, Y_test, test, min_max_scaler = \
    objLSTM.preProcessDataForLSTM(Windfarm2, 1, 'Wind Farm-2')
    
    modelLSTMbi = objLSTM.build_model_LSTMBidirectional(X_train)
    modelLSTMstacked = objLSTM.build_model_LSTMSStacked(X_train)
    global_start_time = time.time()
    
    actualwf1, predictedwf1 = objLSTM.modelLSTMfit(modelLSTMbi, X_train, Y_train,\
                                                   X_test, Y_test, nepochs, 'Wind Farm-2')
    objLSTM.plotGraph(actualwf1, predictedwf1, 'Wind Farm-2', 'Bidirectional LSTM')
    actualwf1, predictedwf1 = objLSTM.modelLSTMfit(modelLSTMstacked, X_train, Y_train,\
                                                   X_test, Y_test, nepochs, 'Wind Farm-1')
    objLSTM.plotGraph(actualwf1, predictedwf1, 'Wind Farm-2', 'Stacked LSTM')
    

    
    #Wind Farm 3 training and prediction
    X_train, Y_train, X_test, Y_test, test, min_max_scaler = \
    objLSTM.preProcessDataForLSTM(Windfarm3, 1, 'Wind Farm-3')
    
    modelLSTMbi = objLSTM.build_model_LSTMBidirectional(X_train)
    modelLSTMstacked = objLSTM.build_model_LSTMSStacked(X_train)
    global_start_time = time.time()
    
    actualwf1, predictedwf1 = objLSTM.modelLSTMfit(modelLSTMbi, X_train, Y_train,\
                                                   X_test, Y_test, nepochs, 'Wind Farm-3')
    objLSTM.plotGraph(actualwf1, predictedwf1, 'Wind Farm-3', 'Bidirectional LSTM')
    actualwf1, predictedwf1 = objLSTM.modelLSTMfit(modelLSTMstacked, X_train, Y_train,\
                                                   X_test, Y_test, nepochs, 'Wind Farm-3')
    objLSTM.plotGraph(actualwf1, predictedwf1, 'Wind Farm-3', 'Stacked LSTM')

    
    #Wind Farm 4 training and prediction
    X_train, Y_train, X_test, Y_test, test, min_max_scaler = \
    objLSTM.preProcessDataForLSTM(Windfarm4, 1, 'Wind Farm-4')
    
    modelLSTMbi = objLSTM.build_model_LSTMBidirectional(X_train)
    modelLSTMstacked = objLSTM.build_model_LSTMSStacked(X_train)
    global_start_time = time.time()
    
    actualwf1, predictedwf1 = objLSTM.modelLSTMfit(modelLSTMbi, X_train, Y_train,\
                                                   X_test, Y_test, nepochs, 'Wind Farm-4')
    objLSTM.plotGraph(actualwf1, predictedwf1, 'Wind Farm-4', 'Bidirectional LSTM')
    actualwf1, predictedwf1 = objLSTM.modelLSTMfit(modelLSTMstacked, X_train, Y_train,\
                                                   X_test, Y_test, nepochs, 'Wind Farm-4')
    objLSTM.plotGraph(actualwf1, predictedwf1, 'Wind Farm-4', 'Stacked LSTM')
    

    #Wind Farm 5 training and prediction
    X_train, Y_train, X_test, Y_test, test, min_max_scaler = \
    objLSTM.preProcessDataForLSTM(Windfarm5, 1, 'Wind Farm-5')
    
    modelLSTMbi = objLSTM.build_model_LSTMBidirectional(X_train)
    modelLSTMstacked = objLSTM.build_model_LSTMSStacked(X_train)
    global_start_time = time.time()
    
    actualwf1, predictedwf1 = objLSTM.modelLSTMfit(modelLSTMbi, X_train, Y_train,\
                                                   X_test, Y_test, nepochs, 'Wind Farm-5')
    objLSTM.plotGraph(actualwf1, predictedwf1, 'Wind Farm-5', 'Bidirectional LSTM')
    actualwf1, predictedwf1 = objLSTM.modelLSTMfit(modelLSTMstacked, X_train, Y_train,\
                                                   X_test, Y_test, nepochs, 'Wind Farm-5')
    objLSTM.plotGraph(actualwf1, predictedwf1, 'Wind Farm-5', 'Stacked LSTM')
    
    
    
