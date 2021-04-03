#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 23:31:01 2020

@author: Mehmood Ali Khan
"""
import numpy as np
import numpy.lib.recfunctions as recfunctions

def dateTimeFunction(datetime, hour_offset):
    year=''
    month=''
    date=''
    hour=''
    year=datetime[0:4]
    month=datetime[4:6]
    date=datetime[6:8]
    hour=datetime[8:10]  
    #print("Year", year )
    #print("Month", month )
    #print("Date", date )
    #print("Hour", hour )  
    if( (int(hour)+int(hour_offset))>23 ):
        hour=str( ( int(hour) + int(hour_offset) ) - 24 )
        days=1
        while( int(hour) >= 24):
            hour=str( int(hour) - 24 )
            days+=1
        if(int(hour)<10):
            hour= '0' + hour
        if( (int(date)+days)>31 and int(month) in [1, 3, 5, 7, 8, 10, 12] ):
            date=str( (int(date)+days) - 31 )
            m=1
            while( int(date)>31 ):
                date=str( int(date) - 31 )
                m+=1
            if(int(date)<10):
                date='0' + date
            
            if( int(month) + m > 12 ):
                month = str( (int(month) + m) - 12 )
                year=str(int(year)+1)
                while(int(month) > 12):
                    month=str(int(month) - 12)
                    year=str(int(year)+1)
                if(int(month) < 10 ):
                    month = '0' + month
            else:
                month=str(int(month)+m)
                if(int(month)<10):
                    month= '0' + month
        elif( (int(date)+days)>30 and int(month) in [4, 6,9,11] ):
            date=str( (int(date)+days) - 30 )
            m=1
            while( int(date)>31 ):
                date=str( int(date) - 30 )
                m+=1
            if(int(date)<10):
                date='0' + date
            if( int(month) + m > 12 ):
                month = str( (int(month) + m) - 12 )
                year=str(int(year)+1)
                while(int(month) > 12):
                    month=str(int(month) - 12)
                    year=str(int(year)+1)
                if(int(month) < 10 ):
                    month = '0' + month
            else:
                month=str(int(month)+m)
                if(int(month)<10):
                    month= '0' + month
        elif( (int(date)+days)>28 and int(month) == 2 ):
            date=str( (int(date)+days) - 28 )
            m=1
            while( int(date)>28 ):
                date=str( int(date) - 28 )
                m+=1
            if(int(date)<10):
                date='0' + date
            if( int(month) + m > 12 ):
                month = str( (int(month) + m) - 12 )
                year=str(int(year)+1)
                while(int(month) > 12):
                    month=str(int(month) - 12)
                    year=str(int(year)+1)
                if(int(month) < 10 ):
                    month = '0' + month
            else:
                month=str(int(month)+m)
                if(int(month)<10):
                    month= '0' + month
        else:
            date=str(int(date)+ days)
            if(int(date)<10):
                date= '0' + date
    else:
        hour=str(int(hour)+int(hour_offset))
        if(int(hour)<10):
            hour= '0' + hour
    dateTimeNew = year + month + date + hour
    #print(datetime," ", hour_offset," ", dateTimeNew)
    return dateTimeNew

class DataPreprocessing:
    def prepareWindFarmDateTimeColumn(self, wf, farmNumber):
        firstIndex=0
        datetimecol = np.empty( (wf.shape[0]),dtype='S10')
        datetimecol[:] = 'dateTime'
        wf = recfunctions.append_fields(wf,'datetime',datetimecol,usemask=False)
        for row in wf:
            row = list(row)
            hour_offset = row[1]
            if(firstIndex==0):
                firstIndex+=1
            else:
                dateTime = dateTimeFunction(row[0], int(hour_offset) )
                wf[firstIndex]['datetime']=dateTime
                firstIndex+=1
        np.savetxt("wf"+str(farmNumber)+"_dateTime_col.csv",wf,fmt='%s,%s,%s,%s,%s,%s,%s')
        return wf
    def combinePowerWithForecastVals(self, train, windFarm, farmNumber):
        dataRowType = np.dtype([ ('dateTime', np.object_), 
                                 ('u', np.object_),
                                 ('v', np.object_),
                                 ('ws', np.object_),
                                 ('wd', np.object_),
                                 ('wp', np.object_) ] )
        windDataHeader = np.array([('dateTime', 'u', 'v', 'ws', 'wd', 'wp')], 
                                   dtype=dataRowType)
        for row in train:
            index = np.where(windFarm[0:]['datetime']==row[0])
            index = np.array(index)
            if(len(index[0])==1):
                dataRow=np.array([(str(windFarm[index[0][0]]['datetime']),
                                   str(windFarm[index[0][0]]['u']),
                                   str(windFarm[index[0][0]]['v']),
                                   str(windFarm[index[0][0]]['ws']),
                                   str(windFarm[index[0][0]]['wd']),
                                   str(row[farmNumber]) )], dtype=dataRowType)
                windDataHeader = np.hstack((windDataHeader,dataRow))
            elif(len(index[0])==2):
                dataRow=np.array([(str(windFarm[int(index[0][0])]['datetime']),
                                   str((float(windFarm[int(index[0][0])]['u'])+
                                        float(windFarm[int(index[0][1])]['u']))/2),
                                   str((float(windFarm[int(index[0][0])]['v'])+
                                        float(windFarm[int(index[0][1])]['v']))/2),
                                   str((float(windFarm[int(index[0][0])]['ws'])+
                                        float(windFarm[int(index[0][1])]['ws']))/2),
                                   str((float(windFarm[int(index[0][0])]['wd'])+
                                        float(windFarm[int(index[0][1])]['wd']))/2),
                                   str(row[farmNumber]) )], dtype=dataRowType)
                windDataHeader = np.hstack((windDataHeader,dataRow))
            elif(len(index[0])==3):
                dataRow=np.array([(str(windFarm[int(index[0][0])]['datetime']),
                                   str((float(windFarm[int(index[0][0])]['u'])+
                                        float(windFarm[int(index[0][1])]['u'])+
                                        float(windFarm[int(index[0][2])]['u']))/3),
                                   str((float(windFarm[int(index[0][0])]['v'])+
                                        float(windFarm[int(index[0][1])]['v'])+
                                        float(windFarm[int(index[0][2])]['v']))/3),
                                   str((float(windFarm[int(index[0][0])]['ws'])+
                                        float(windFarm[int(index[0][1])]['ws'])+
                                        float(windFarm[int(index[0][2])]['ws']))/3), 
                                   str((float(windFarm[int(index[0][0])]['wd'])+
                                        float(windFarm[int(index[0][1])]['wd'])+
                                        float(windFarm[int(index[0][2])]['wd']))/3),
                                   str(row[farmNumber]) )], dtype=dataRowType)
                windDataHeader = np.hstack((windDataHeader,dataRow))
            elif(len(index[0])==4):
                dataRow=np.array([(str(windFarm[int(index[0][0])]['datetime']), 
                                   str((float(windFarm[int(index[0][0])]['u'])+
                                        float(windFarm[int(index[0][1])]['u'])+
                                        float(windFarm[int(index[0][2])]['u'])+
                                        float(windFarm[int(index[0][3])]['u']))/4),
                                   str((float(windFarm[int(index[0][0])]['v'])+
                                        float(windFarm[int(index[0][1])]['v'])+
                                        float(windFarm[int(index[0][2])]['v'])+
                                        float(windFarm[int(index[0][3])]['v']))/4), 
                                   str((float(windFarm[int(index[0][0])]['ws'])+
                                        float(windFarm[int(index[0][1])]['ws'])+
                                        float(windFarm[int(index[0][2])]['ws'])+
                                        float(windFarm[int(index[0][3])]['ws']))/4), 
                                   str((float(windFarm[int(index[0][0])]['wd'])+
                                        float(windFarm[int(index[0][1])]['wd'])+
                                        float(windFarm[int(index[0][2])]['wd'])+
                                        float(windFarm[int(index[0][3])]['wd']))/4),
                                   str(row[farmNumber]) )], dtype=dataRowType)
                windDataHeader = np.hstack((windDataHeader,dataRow))
                #print("index No:",index[0][0], "No. of Indexes returned:", len(index[0]), "index Values:", int(index[0][0]), int(index[0][1]), int(index[0][2]), int(index[0][3]))
                #print("Wind Data Shape: ", windDataHeader.shape)
                #print("Wind Data[0]:\n", windDataHeader[0])
                #print("Wind Data[1]:\n", windDataHeader[1])
                #print("Wind Data:\n", windDataHeader)
                #break
        print("Returning Data.. Exiting from function!")
        return windDataHeader     
    def accumulatePowerAndForecastVals(self):
        return 0
    def accumulateAllWindFarmsData(self, allWindFarmsList):
        compDataSet = np.concatenate((allWindFarmsList[0],allWindFarmsList[1][1:],
                                      allWindFarmsList[2][1:],allWindFarmsList[3][1:],
                                      allWindFarmsList[4][1:],allWindFarmsList[5][1:],
                                      allWindFarmsList[6][1:]))
        return compDataSet
    def saveDataSetInCSVFormat(self, DataSet):
        np.savetxt("completeWfDatawithWpVals.csv",DataSet,fmt='%s,%s,%s,%s,%s,%s,%s')