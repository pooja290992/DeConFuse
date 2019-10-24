import pandas as pd
import numpy as np 

import os 
from datetime import date
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.modules.pooling import *
import time
import random
import datetime
#import torchvision
#import torchvision.transforms as transforms
from torch.autograd import Variable
import tensorflow as tf
from sklearn import preprocessing
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


#base_path = '/home/pooja/Desktop/IIITD/Research/Stock_class_classification/'
base_path = '../'

#loads the data 
def getData(fileName):
    filepath = base_path + 'data/' + fileName
    data = pd.read_csv(filepath,sep = ',')
    if fileName == 'all_stocks_2006-01-01_to_2018-01-01.csv':
        data.rename(columns = {'Date':'DATE','Low':'LOW','High':'HIGH','Close':'CLOSE','Open':'OPEN','Volume':'CONTRACTS','Name':'SYMBOL'},inplace=True)
    ##display('data\'s shape : ',data.shape)
    display(data.head())
    ##display(data.describe())
    return data
    
    
def labelData(data):
    labels = np.where(data['CLOSE'] - data['OPEN']>0,1,0)
    data['Stock_class'] = labels
    ##print(type(labels),labels.shape)
    ##print(data.shape)
    labels = pd.DataFrame(labels,columns= ['Stock_class'])    
    ##print(labels.head())
    ##print(labels.shape)
    ##print(data.head())
    ##display(labels.describe())
    return data,labels


def labelDataReg(data):
    labels = np.where(data['CLOSE'] - data['OPEN']>0,1,0)
    data['Stock_class'] = labels
    ##print(type(labels),labels.shape)
    ##print(data.shape)
    labels = pd.DataFrame(labels,columns= ['Stock_class'])    
    ##print(labels.head())
    ##print(labels.shape)
    ##print(data.head())
    ##display(labels.describe())
    return data,labels
    
    
def splitData(data,labels,test_size = 0.20):
    train_size = int(len(data)*(1-test_size))
    X_train, X_test = data[0:train_size],data[train_size:]  
    Y_train, Y_test = labels[0:train_size],labels[train_size:]  
    return X_train,Y_train,X_test,Y_test
    
    
    
def splitDataWithVal(data,labels,test_size = 0.20,val_size = 0.25):
    train_size = int(len(data)*(1-test_size))
    val_size = int(train_size*(1-val_size))
    X_train, X_val, X_test = data[0:val_size],data[val_size:train_size],data[train_size:]  
    Y_train, Y_val, Y_test = labels[0:val_size],labels[val_size:train_size],labels[train_size:]
    return X_train,Y_train,X_val,Y_val,X_test,Y_test
    
    
def getWindowedData(data_df,group_name = 'ABIRLANUVO',window_size = 7,features_list = ['CLOSE','OPEN','HIGH','LOW','CONTRACTS','DATE','Stock_class']):
    g = data_df.groupby('SYMBOL')
    g.groups.keys()
    X = g.get_group(group_name)
    X.sort_values(by = ['DATE'],inplace=True)
    rows,cols=X.shape
    #display(X.tail())
    #display(type(X[0:10]))#.shape)
    Stock_class=X[['Stock_class']].copy()
    Stock_class=np.asarray(Stock_class)
    stock_class=Stock_class.reshape(1,rows)
    ##print(stock_class.shape)
    labels_new = stock_class[0,window_size:]
    ##print(labels_new.shape[0])
    windowed_data = []
    stock_table = []
    start_idx = 0
    end_idx = 0
    for i in range(labels_new.shape[0]):
        start_idx = i 
        end_idx = start_idx + window_size
        if end_idx < rows:
            windowed_data.append(X[start_idx:end_idx])
            stock_table.extend(X[end_idx:end_idx+1][features_list].values.tolist())
    ##print('len of windowed_data:',len(windowed_data))
    ##print('type of windowed_data 0th element is :', type(windowed_data[0]))
    ##print('shape of windowed_data 0th element is :', windowed_data[0].shape)
    ##print('windowed data ')
    return labels_new,windowed_data,stock_table
    


def getWindowedDataReg(data_df,group_name = 'ABIRLANUVO',window_size = 7,features_list = ['CLOSE','OPEN','HIGH','LOW','CONTRACTS','DATE','Stock_class']):
    g = data_df.groupby('SYMBOL')
    g.groups.keys()
    X = g.get_group(group_name)
    X.sort_values(by = ['DATE'],inplace=True)
    rows,cols=X.shape
    #display(X.tail())
    #display(type(X[0:10]))#.shape)
    Stock_class=X[['Stock_class']].copy()
    Stock_class=np.asarray(Stock_class)
    stock_class=Stock_class.reshape(1,rows)
    ##print(stock_class.shape)
    labels_new = stock_class[0,window_size:]
    temp = X.copy()
    temp = temp.reset_index()
    temp2 = temp[features_list[0:5]]
    #print(temp2.head(12))
    #print(type(temp2.values))
    #print(temp2.values.shape)
    #vals = temp2.values
    #print(vals[window_size:])
    next_day_values = temp2.values[window_size:]
    ##print(labels_new.shape[0])
    windowed_data = []
    stock_table = []
    future_prices = []
    start_idx = 0
    end_idx = 0
    for i in range(labels_new.shape[0]):
        start_idx = i 
        end_idx = start_idx + window_size
        if end_idx < rows:
            windowed_data.append(X[start_idx:end_idx])
            stock_table.extend(X[end_idx:end_idx+1][features_list].values.tolist())
            future_prices
    ##print('len of windowed_data:',len(windowed_data))
    ##print('type of windowed_data 0th element is :', type(windowed_data[0]))
    ##print('shape of windowed_data 0th element is :', windowed_data[0].shape)
    ##print('windowed data ')
    return labels_new,windowed_data,stock_table, next_day_values
    
    
    
    
def getFeatWiseData(windowed_data,features_list):
    temp_data = {}
    for feat in features_list: 
        temp_data[feat] = []
    ##print(temp_data)
    #print(windowed_data[0]['CLOSE'].values.tolist())
    for i in range(len(windowed_data)):
        temp = windowed_data[i]
        for feat in features_list: 
            temp_data[feat].append(temp[feat])
    #print('len of close series : ',len(temp_data['CLOSE']))
    keys = list(temp_data.keys())
    #print(keys)
    temp_records =  np.asarray(temp_data[keys[0]])
    for key in keys[1:]:
        temp = np.asarray(temp_data[key])
        temp_records = np.dstack((temp_records,temp))
    ##print(temp_records.shape)
    return temp_records


#def getLastRecord(stock):
    
    

def getPrevDayFeatures(feat_wise_data):
    prev_day_prices = []
    for record in feat_wise_data:
        prev_day_prices.append(record[-1,:])
    prev_day_prices = np.asarray(prev_day_prices)
    return prev_day_prices

    
    
def toFloatTensor(numpy_array):
    # Numpy array -> Tensor
    return torch.from_numpy(numpy_array).float()
    
    
   

class FinancialData(Dataset):
    def __init__(self,train,labels):
        self.train_data=train
        self.labels=labels
        self.total_samples =len(self.train_data)
        
    def __len__(self):
        return len(self.train_data)
    
    def __getitem__(self,idx):
        
        return toFloatTensor(self.train_data[idx,:,:].T), self.labels[idx]
    
    
    
class RegFinancialData(Dataset):
    def __init__(self,train,future_prices):
        self.train_data = train
        self.future_prices = future_prices
        self.total_samples =len(self.train_data)
        
    def __len__(self):
        return len(self.train_data)
    
    def __getitem__(self,idx):
        
        return toFloatTensor(self.train_data[idx,:,:].T),  self.future_prices[idx]
