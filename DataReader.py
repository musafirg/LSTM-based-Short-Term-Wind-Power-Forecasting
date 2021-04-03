#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 23:41:22 2020

@author: Mehmood Ali Khan
"""
import os
import csv
import numpy as np

class DataReader:
    def readData(self):
        print('Initiated reading data files from the following path: \n'+ 
              os.getcwd()+" "+'(The Current Working Directory)\n')
        train = np.genfromtxt(open(r"train.csv", "r"), delimiter=",", 
                converters ={0: str}, 
                dtype='object, object, object, object, object, object, object, object',
                names = ["date", "wp1", "wp2","wp3","wp4","wp5","wp6","wp7"] )
        print('train.csv file loaded successfully.\n')
        wf1 = np.genfromtxt(open(r"windforecasts_wf1.csv", "r"), delimiter=",", 
                converters ={0: str}, 
                dtype='object, object, object, object, object, object',
                names = ["date", "hours", "u","v","ws","wd"] )
        print('wf1.csv file loaded successfully.\n')
        wf2 = np.genfromtxt(open(r"windforecasts_wf2.csv", "r"), delimiter=",", 
                converters ={0: str}, 
                dtype='object, object, object, object, object, object',
                names = ["date", "hours", "u","v","ws","wd"] )
        print('wf2.csv file loaded successfully.\n')
        wf3 = np.genfromtxt(open(r"windforecasts_wf3.csv", "r"), delimiter=",", 
                converters ={0: str}, 
                dtype='object, object, object, object, object, object',
                names = ["date", "hours", "u","v","ws","wd"] )
        print('wf3.csv file loaded successfully.\n')
        wf4 = np.genfromtxt(open(r"windforecasts_wf4.csv", "r"), delimiter=",", 
                converters ={0: str}, 
                dtype='object, object, object, object, object, object',
                names = ["date", "hours", "u","v","ws","wd"] )
        print('wf4.csv file loaded successfully.\n')
        wf5 = np.genfromtxt(open(r"windforecasts_wf5.csv", "r"), delimiter=",", 
                converters ={0: str}, 
                dtype='object, object, object, object, object, object',
                names = ["date", "hours", "u","v","ws","wd"] )
        print('wf5.csv file loaded successfully.\n')
        return train, wf1, wf2, wf3, wf4, wf5 #, wf6, wf7
    
    def readTrainingData(self, fileName):
        train = np.genfromtxt(open(fileName, "r"), delimiter=",", 
                converters ={0: str}, 
                dtype='object, object, object, object, object, object, object, object',
                names = ["date", "wp1", "wp2","wp3","wp4","wp5","wp6","wp7"] )
        return train
    
    def readAccumulatedDataSet(self, fileName):
        print('\nInitiated reading Specified Dataset file from the following path: \n\n'+ 
              os.getcwd()+" "+'(The Current Working Directory)\n')
        dataSet = np.genfromtxt(open(fileName, "r"), delimiter=",", 
                converters ={0: str}, 
                dtype='object, object, object, object, object, object, object',
                names = ["date", "hours", "u","v","ws","wd","wp"] )
        print('Complete DataSet file loaded successfully.\n')
        return dataSet
    
    def readWindFarm(self, fileName):
        windFarmData = np.genfromtxt(open(fileName, "r"), delimiter=",", 
                converters ={0: str}, 
                dtype='object, object, object, object, object, object, object',
                names = ["date", "hours", "u","v","ws","wd","datetime"] )
        return windFarmData

    def readWindFarmWithPowerVals(self, fileName):
        windFarmData = np.genfromtxt(open(fileName, "r"), delimiter=",", 
                converters ={0: str}, 
                dtype='object, object, object, object, object, object',
                names = ["dateTime","u","v","ws","wd","wp"] )
        return windFarmData

    
    def readCompleteDataSet(self, filename):
        print('\nInitiated reading Specified Dataset file from the following path: \n\n'
              + os.getcwd()+" "+'(The Current Working Directory)\n')
        data = []
        checkForFirstLine=0
        with open(filename) as f:
            reader = csv.reader(f, delimiter=',', skipinitialspace=True)
            for line in reader:
                if(checkForFirstLine==0):
                    checkForFirstLine=1
                    continue
                else:
                    try:
                        #print(line)
                        date, hours,(u, v, ws, wd, wp) = line[0], int(line[1]),map(float, line[2:] )
                        #(date, hours, u, v, ws, wd, wp) = (line[0], int(line[1]), float(line[2]), float(line[3]), float(line[4]), float(line[5]), float(line[6]) )
                    except ValueError as e:
                        #print('Skipping line: %s [because of: %s]' % (line, e))
                        continue
                    data.append([date, hours, u, v, ws, wd, wp])
        return data        
