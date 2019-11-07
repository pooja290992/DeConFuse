#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from torch.autograd import Variable
import tensorflow as tf
from sklearn import preprocessing
import matplotlib.pyplot as plt
from utils import *
from data_processing import * 
import scipy as sp 
from sklearn.metrics import *
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier


# # Transform Learning

# In[2]:


def calOutShape(input_shape,ksize1=3,ksize2 =3,stride=1,maxpool1=False, maxpool2=False, mpl_ksize=2):
    mpl_stride = 2
    pad = ksize1//2
    dim2 = int((input_shape[2]-ksize1+2*pad)/stride) + 1
    if maxpool1 == True:
        dim2 = (dim2 - mpl_ksize)//mpl_stride + 1
    pad = ksize2//2
    dim2 = int((dim2-ksize2+2*pad)/stride) + 1
    if maxpool2 == True:
        dim2 = (dim2 - mpl_ksize)//mpl_stride + 1
    return dim2



class Transform(nn.Module):
    
    def __init__(self,input_shape, out_planes1 = 8, out_planes2 = 16,ksize1 = 3,ksize2 = 3,maxpool1 = False, 
                 maxpool2 = False,mpl_ksize=2):
        super(Transform, self).__init__()
        self.ksize1 = ksize1
        self.ksize2 = ksize2
        self.mpl_ksize = mpl_ksize
        self.out_planes1 = out_planes1
        self.out_planes2 = out_planes2
        self.init_T()
        self.maxpool1 = maxpool1
        self.maxpool2 = maxpool2
        self.input_shape = input_shape
        self.i = 1
        self.atom_ratio = 0.5
        self.init_X()
        self.gap = AdaptiveAvgPool1d(1)
        
        
        
    
    def init_T(self):
        conv = nn.Conv1d(1, out_channels = self.out_planes1, kernel_size = self.ksize1, stride=1, bias=True)
        self.T1 = conv._parameters['weight']
        conv = nn.Conv1d(in_channels = self.out_planes1, out_channels = self.out_planes2, 
                         kernel_size = self.ksize2, stride=1, bias=True)
        self.T2 = conv._parameters['weight']

       
           
        
    def init_X(self):
        dim2 = calOutShape(self.input_shape,self.ksize1,self.ksize2,stride = 1,maxpool1 = self.maxpool1, 
                           maxpool2 = self.maxpool2, mpl_ksize = self.mpl_ksize)
        X_shape = [self.input_shape[0],self.out_planes2,dim2]
        self.X  = nn.Parameter(torch.randn(X_shape), requires_grad=True)
        self.num_features = self.out_planes2*dim2
        self.num_atoms = int(self.num_features*self.atom_ratio*5) #dim2//2
        T_shape = [self.num_atoms,self.num_features]
        self.T = nn.Parameter(torch.randn(T_shape), requires_grad=True)

        
    def forward(self, inputs):
        x = F.conv1d(inputs, self.T1,padding = self.ksize1//2)
        if self.maxpool1:
            x = F.max_pool1d(x, 2)
        x = F.selu(x)
        x = F.conv1d(x, self.T2, padding = self.ksize2//2)
        if self.maxpool2:
            x = F.max_pool1d(x, 2)
        y = torch.mm(self.T,x.view(x.shape[0],-1).t())
        return x, y
        
          
    def get_params(self):
        return self.T1, self.T2, self.X, self.T
    
    
    def X_step(self):
        self.X.data = torch.clamp(self.X.data, min=0)


    def Z_step(self):
        self.Z.data = torch.clamp(self.Z.data, min=0)
        
        
    def get_TZ_Dims(self):
        return self.num_features,self.num_atoms, self.input_shape[0]
        
        
class Network(nn.Module): 
    def __init__(self,inputs_shape=(4,5,1),out_planes1 = 8, out_planes2 = 16,ksize1 = 3,ksize2 = 3,
             maxpool1=False, maxpool2=False, mpl_ksize=2,num_classes=2):
        super(Network, self).__init__()
        self.Transform1 = Transform(inputs_shape,out_planes1 = out_planes1, out_planes2 = out_planes2,ksize1 = ksize1,
                                    ksize2 = ksize2,maxpool1=maxpool1, maxpool2=maxpool2, mpl_ksize=mpl_ksize)
        self.Transform2 = Transform(inputs_shape,out_planes1 = out_planes1, out_planes2 = out_planes2,ksize1 = ksize1,
                                    ksize2 = ksize2,maxpool1=maxpool1, maxpool2=maxpool2, mpl_ksize=mpl_ksize)
        self.Transform3 = Transform(inputs_shape,out_planes1 = out_planes1, out_planes2 = out_planes2,ksize1 = ksize1,
                                    ksize2 = ksize2,maxpool1=maxpool1, maxpool2=maxpool2, mpl_ksize=mpl_ksize)
        self.Transform4 = Transform(inputs_shape,out_planes1 = out_planes1, out_planes2 = out_planes2,ksize1 = ksize1,
                                    ksize2 = ksize2,maxpool1=maxpool1, maxpool2=maxpool2, mpl_ksize=mpl_ksize)
        self.Transform5 = Transform(inputs_shape,out_planes1 = out_planes1, out_planes2 = out_planes2,ksize1 = ksize1,
                                    ksize2 = ksize2,maxpool1=maxpool1, maxpool2=maxpool2, mpl_ksize=mpl_ksize)
        self.num_features,self.num_atoms, self.input_shape = self.Transform1.get_TZ_Dims()
        Z_shape = [self.num_atoms,self.input_shape]
        self.Z = nn.Parameter(torch.randn(Z_shape), requires_grad=True)
        self.pred_list = []
        self.init_TX()
        

    def init_TX(self):
        self.T11,self.T21, self.X1, self.Tp1 = self.Transform1.get_params()
        self.T12,self.T22, self.X2, self.Tp2 = self.Transform2.get_params()
        self.T13,self.T23, self.X3, self.Tp3 = self.Transform3.get_params()
        self.T14,self.T24, self.X4, self.Tp4 = self.Transform4.get_params()
        self.T15,self.T25, self.X5, self.Tp5 = self.Transform5.get_params()
        self.T1 = torch.stack((self.T11,self.T12,self.T13,self.T14,self.T15),1)
        self.T2 = torch.stack((self.T21,self.T22,self.T23,self.T24,self.T25),1)
        self.X = torch.stack((self.X1,self.X2,self.X3,self.X4,self.X5),1) 
        self.T = torch.stack((self.Tp1,self.Tp2,self.Tp3,self.Tp4,self.Tp5),1) 
        
        
        
    def forward(self,x):
        batch_size, no_of_series, no_of_days = x.shape
        
        close = np.reshape(x[:,0],(batch_size,1,no_of_days))
        out1,out1p = self.Transform1(close)
        
        _open = np.reshape(x[:,1],(batch_size,1,no_of_days))
        out2,out2p = self.Transform2(_open)
        
        
        high = np.reshape(x[:,2],(batch_size,1,no_of_days))
        out3,out3p = self.Transform3(high)
        
        low = np.reshape(x[:,3],(batch_size,1,no_of_days))
        out4,out4p = self.Transform4(low)
        
        volume = np.reshape(x[:,4],(batch_size,1,no_of_days))
        out5, out5p = self.Transform5(volume)
        
        self.pred_list = [out1,out2,out3,out4,out5]

        gp1 = out1p + out2p + out3p + out4p + out5p
        return gp1
    
    
    def X_step(self):
        self.Transform1.X_step()
        self.Transform2.X_step()
        self.Transform3.X_step()
        self.Transform4.X_step()
        self.Transform5.X_step()
        
        
    def Z_step(self):
        self.Z.data = torch.clamp(self.Z.data, min=0)
        
    
    def conv_loss_distance(self):
        self.init_TX()
        
        loss = 0.0
        X_list = [self.X1,self.X2,self.X3,self.X4,self.X5]
        for i in range(len(self.pred_list)): 
            X = X_list[i].view(X_list[i].size(0), -1)
            predictions = self.pred_list[i].view(self.pred_list[i].size(0), -1)
            Y = predictions - X
            loss += Y.pow(2).mean()
            
        return loss
    
        
    def conv_loss_logdet(self):

        loss = 0.0
        for T in [self.T11,self.T21,self.T12,self.T22,self.T13,self.T23,self.T14,self.T24,self.T15,self.T25]:
            T = T.view(T.shape[0],-1)
            U, s, V = torch.svd(T)
            loss += -s.log().sum()
        return loss
        
        
    def conv_loss_frobenius(self):
        loss = 0.0
        for T in [self.T11,self.T21,self.T12,self.T22,self.T13,self.T23,self.T14,self.T24,self.T15,self.T25]:
            loss += T.pow(2).sum()
        return loss
    

    def loss_distance(self,predictions):

        loss = 0.0
        Y = predictions - self.Z
        loss += Y.pow(2).mean()    
        
        return loss
        
    def loss_logdet(self):
        loss = 0.0
        T = torch.stack((self.Tp1,self.Tp2,self.Tp3,self.Tp4,self.Tp5),1)
        T = T.view(T.shape[0],-1)
        U, s, V = torch.svd(T)
        loss = -s.log().sum()
        return loss
        
        
    def loss_frobenius(self):
        loss = 0.0
        t_p = torch.stack((self.Tp1,self.Tp2,self.Tp3,self.Tp4,self.Tp5),1)
        loss = t_p.pow(2).sum()
        return loss


    def computeLoss(self,predictions,mu,eps,lam):
        loss1 = self.conv_loss_distance()
        loss2 = self.conv_loss_frobenius() * eps
        loss3 = self.conv_loss_logdet() * mu
        loss4 = self.loss_distance(predictions)
        loss5 = self.loss_frobenius() * eps
        loss6 = self.loss_logdet() * mu
        loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
        return loss

    
    def getTZ(self):
        return self.T.view(self.T.shape[0],-1), self.Z


# ## Training Related Functions 

# In[3]:


def train_model(epoch, model, optimizer, train_loader, batch_size, mu, eps, lam):
    model.train()
    t0 = time.time()
    correct = 0
    total = 0
    final_loss = 0
    i = 0 
    j = 0
    T_list = []
    Z_list = []
    for batch_idx, (X,future_prices) in enumerate(train_loader):
        #print('batch:',batch_idx)
        data,future_prices = map(lambda x: Variable(x), [X,future_prices])
        data_size1 = data.shape[0]
        if j == 0: 
            prev_data = data
            prev_future_prices = future_prices
            j += 1
        if data.shape[0]<batch_size:
            diff = batch_size - data.shape[0]
            temp_data,temp_labels = prev_data[-diff:,:,:], prev_future_prices[-diff:]
            i = 1
            data, temp_future_prices = torch.cat((data,temp_data),0),torch.cat((future_prices,temp_future_prices),0)
            print('appended data')
            
        optimizer.zero_grad()
        

        output = model(data)

        
        final_output = output
        
        loss = model.computeLoss(final_output,mu,eps,lam)
        final_loss += loss
        loss.backward()
        optimizer.step()
        model.X_step()
        model.Z_step()
        prev_data = data
        prev_future_prices = future_prices
    print('Epoch : {} , Training loss : {:.4f}\n'.format(epoch, final_loss.item()))
    return final_loss.item()

 
    
def train_on_batch(lr,epochs,momentum,X_train,Y_train,X_test,Y_test,batch_size):
    print('seed:',seed)
    cuda = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_loader = DataLoader(RegFinancialData(X_train,Y_train),batch_size=batch_size,shuffle=True) 
    test_loader = DataLoader(RegFinancialData(X_test,Y_test),batch_size=batch_size,shuffle=False) 
    
    
    mu = 0.01
    eps = 0.0001
    lam = 0 
    out_planes1 = out_pl1
    out_planes2 = out_pl2
    ksize1 = ks1
    ksize2 = ks2
    maxpool1 = maxpl1
    maxpool2 = maxpl2
    mpl_ksize = mpl_ks
    model = Network(inputs_shape=(batch_size,1,window_size),out_planes1 = out_planes1,out_planes2 = out_planes2,  
                    ksize1 = ksize1,ksize2 = ksize2, maxpool1 = maxpool1, maxpool2 = maxpool2, mpl_ksize=mpl_ksize)
#     for params in model.parameters():
#         print(params)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-5, 
                                 amsgrad=False)

    for epoch in range(1, epochs + 1):
        tr_loss = train_model(epoch, model, optimizer, train_loader, batch_size, mu, eps, lam)
        if epoch%plot_epoch_interval==0:
            train_loss.append(tr_loss)
            epochs_list.append(epoch)
    model.eval()

    S_train = Variable(torch.from_numpy(X_train).float(), requires_grad=False)
    S_test  = Variable(torch.from_numpy(X_test).float(), requires_grad=False)
    Z_train =  model(S_train).cpu().data.numpy()
    Z_test  = model(S_test).cpu().data.numpy()
    print('*'*100)
    print("Shape of Z_train: " + str(Z_train.shape))
    print("Shape of Z_test:  " + str(Z_test.shape))
    print('*'*100)

    return Z_train.transpose(),Z_test.transpose(),train_loss


# # Main

# In[4]:


window_size = 5
fileName = 'phd_research_data.csv'
data_df = getData(fileName)
if fileName == 'phd_research_data.csv':
    data_df.drop(['Unnamed: 0'],inplace=True,axis=1)
#we are not using these labels
data_df,_ = labelData(data_df.copy())
data = np.asarray(data_df)


# In[5]:


stocks_list = getStocksList(data_df)


# In[6]:


def checkClassImbal(Y_train):
    Ytrain_df= pd.DataFrame(Y_train,columns=[0])
    print(Ytrain_df.shape)
    print(Ytrain_df.columns)
    df = Ytrain_df.groupby(0).size()
    print(df)
    return df


# In[7]:


start = 0
end = 150
seed_range = 10


# In[8]:


train_loss = []
train_accuracies = []
epochs_list = []
learning_rates = []
epoch_interval = 10
plot_epoch_interval = 5
test_accuracies = []


# In[9]:


lr = 0.001
momentum = 0.9
epochs = 100
test_size = 0.2
features_list = ['CLOSE','OPEN','HIGH','LOW','CONTRACTS']


# In[12]:


out_pl1 = 4
out_pl2 = 8
maxpl1 = True
maxpl2 = False
ks1 = 5
ks2 = 3
mpl_ks = 2
custom_batch_size_flag = False
bs = 32
if custom_batch_size_flag == True:
    param_path = '_op1_' + str(out_pl1) + '_op2_' + str(out_pl2) +'_mp1_' + str(maxpl1) + '_mp2_' + str(maxpl2) + '_ks1_' + str(ks1) + '_ks2_' + str(ks2)             + '_bs_' + str(bs) + '_new'
else:
    param_path = '_op1_' + str(out_pl1) + '_op2_' + str(out_pl2) +'_mp1_' + str(maxpl1) + '_mp2_' + str(maxpl2) + '_ks1_' + str(ks1) + '_ks2_' + str(ks2) + '_new'
print(param_path)


# In[13]:


t_0 = time.time()
for stock in stocks_list[start:end]:
    t0 = time.time()
    _,windowed_data,_, next_day_values = getWindowedDataReg(data_df,stock,window_size)
    feat_wise_data = getFeatWiseData(windowed_data,features_list)
    feat_wise_data = feat_wise_data[:feat_wise_data.shape[0]-1]
    prev_day_values = getPrevDayFeatures(feat_wise_data)
    next_day_values = next_day_values[:,0]
    next_day_values = next_day_values[0:next_day_values.shape[0]-1]
    X_train,Y_train,X_test,Y_test = splitData(feat_wise_data,next_day_values,test_size=test_size)
    prev_day_values = prev_day_values[X_train.shape[0]:][:,0]
    print('prev_day_values.shape:',prev_day_values.shape)
    print('X_test.shape:',X_test.shape)
    print('Y_test.shape:',Y_test.shape)
    print('next_day_values.shape:', next_day_values.shape)
    prev_val_path = base_path + 'data/Reg2/TL_Test/' + stock + param_path +  '_' + str(test_size) + '_tl_yprev_cp.npy'
    np.save(prev_val_path,prev_day_values)
    for sd in range(1,seed_range+1):
        t01 = time.time()
        seed = sd
        print('Training for stock :',stock)
        print('seed : ',seed)
        print('starting at time:',t0)
        print('*'*100)
        if custom_batch_size_flag:
            batch_size = bs
        else:
            batch_size = X_train.shape[0]
        Ztrain, Ztest,train_loss = train_on_batch(lr,epochs,momentum,X_train,Y_train,X_test,Y_test,batch_size)
        xtr_path = base_path + 'data/Reg2/TL_Train/' + stock + param_path +'_' + str(test_size) + '_tl_xtrain' + str(seed) + '.npy'
        ytr_path = base_path + 'data/Reg2/TL_Train/' + stock + param_path +  '_' + str(test_size) + '_tl_ytrain' + str(seed) + '.npy'
        xte_path = base_path + 'data/Reg2/TL_Test/' + stock + param_path +  '_' + str(test_size) + '_tl_xtest' + str(seed) + '.npy'
        yte_path = base_path + 'data/Reg2/TL_Test/' + stock + param_path +  '_' + str(test_size) + '_tl_ytest' + str(seed) + '.npy'
        np.save(xtr_path,Ztrain)
        np.save(ytr_path,Y_train)
        np.save(xte_path,Ztest)
        np.save(yte_path,Y_test)
        t11 = time.time()
        print('*'*100)
        #print('*'*100)
        print('\n')
        print('time taken for training one stock:',datetime.timedelta(seconds = t11-t01))
    t1 = time.time()
    print('time taken for one stock through all seeds',datetime.timedelta(seconds = t1-t0))
t_1 = time.time()
print('time taken for stocks through all seeds : ',str(datetime.timedelta(seconds = t_1-t_0)))


# # External Regressor 

# In[14]:



def ridge_regressor(Xtrain, Ytrain, Xtest, Ytest, alpha = 1.0, random_state = 1):
    clf = Ridge(alpha=alpha,random_state = random_state)
    clf.fit(Xtrain, Ytrain) 
    y_pred = clf.predict(Xtest)
    return y_pred


# In[26]:


res_file_name = base_path+'Results2/Reg2/res_reg_measures.csv'
pred_file_name = base_path+'Results2/Reg2/res_reg_pred.csv'
if os.path.exists(res_file_name):
    os.remove(res_file_name)
if os.path.exists(pred_file_name):
    os.remove(pred_file_name)


# In[27]:


alpha = 0.1
random_state = 1
test_pred_dict = {}
test_measures_dict = {}
cnt = 0
t0 = time.time()
df_temp1 = pd.read_csv(base_path+'data/init.csv') 
for stock in stocks_list[start:end]:
    t01 = time.time()
    print('cnt:',cnt)
    print('stock:',stock)
    test_pred_dict[stock] = {}
    test_measures_dict[stock] = {}
    _, _, _, next_day_values = getWindowedDataReg(data_df,stock,window_size)
    
    next_day_values = next_day_values[0:next_day_values.shape[0]-1]
    seed = int(df_temp1.loc[df_temp1['index']==stock]['seed'].values.tolist()[0])
    xtr_path = base_path + 'data/Reg2/TL_Train/' + stock + param_path +'_' + str(test_size) + '_tl_xtrain' + str(seed) + '.npy'
    xte_path = base_path + 'data/Reg2/TL_Test/' + stock + param_path +  '_' + str(test_size) + '_tl_xtest' + str(seed) + '.npy'
    Ztrain = np.load(xtr_path)
    Ztest = np.load(xte_path)
    #for i in range(1,5):
    limit = Ztrain.shape[0]
    
    next_close_prices, next_open_prices, next_high_prices, next_low_prices, next_day_volume = next_day_values[:,0], next_day_values[:,1],next_day_values[:,2],                                                                        next_day_values[:,3], next_day_values[:,4]
    Y_train_cp = next_close_prices[0:limit] 
    Y_test_cp = next_close_prices[limit:]
    y_pred_cp = ridge_regressor(Ztrain, Y_train_cp, Ztest, Y_test_cp, alpha = alpha, random_state = random_state)
    #weighted MAE 
    mae_cp = np.sum(np.abs(y_pred_cp - Y_test_cp))/np.sum(Y_test_cp)
    test_pred_dict[stock]['True_CP'] = Y_test_cp
    test_pred_dict[stock]['Predicted_CP'] = y_pred_cp
    test_measures_dict[stock]['MAE_CP'] = mae_cp
    
    Y_train_op = next_open_prices[0:limit] 
    Y_test_op = next_open_prices[limit:]
    y_pred_op = ridge_regressor(Ztrain, Y_train_op, Ztest, Y_test_op, alpha = alpha, random_state = random_state)
    mae_op = np.sum(np.abs(y_pred_op - Y_test_op))/np.sum(Y_test_op)
    test_pred_dict[stock]['True_OP'] = Y_test_op
    test_pred_dict[stock]['Predicted_OP'] = y_pred_op
    test_measures_dict[stock]['MAE_OP'] = mae_op
    
    
    Y_train_h = next_high_prices[0:limit] 
    Y_test_h = next_high_prices[limit:]
    y_pred_h = ridge_regressor(Ztrain, Y_train_h, Ztest, Y_test_h, alpha = alpha, random_state = random_state)
    mae_h = np.sum(np.abs(y_pred_h - Y_test_h))/np.sum(Y_test_h)
    test_pred_dict[stock]['True_HP'] = Y_test_h
    test_pred_dict[stock]['Predicted_HP'] = y_pred_h
    test_measures_dict[stock]['MAE_HP'] = mae_h
    
    Y_train_l = next_low_prices[0:limit] 
    Y_test_l = next_low_prices[limit:]
    y_pred_l = ridge_regressor(Ztrain, Y_train_l, Ztest, Y_test_l, alpha = alpha, random_state = random_state)
    mae_l = np.sum(np.abs(y_pred_l - Y_test_l))/np.sum(Y_test_l)
    test_pred_dict[stock]['True_LP'] = Y_test_l
    test_pred_dict[stock]['Predicted_LP'] = y_pred_l
    test_measures_dict[stock]['MAE_LP'] = mae_l
    
    Y_train_vol = next_day_volume[0:limit] 
    Y_test_vol = next_day_volume[limit:]
    y_pred_vol = ridge_regressor(Ztrain, Y_train_vol, Ztest, Y_test_vol, alpha = alpha, random_state = random_state)
    mae_vol = np.sum(np.abs(y_pred_vol - Y_test_vol))/np.sum(Y_test_vol)
    test_pred_dict[stock]['True_Vol'] = Y_test_vol
    test_pred_dict[stock]['Predicted_Vol'] = y_pred_vol
    test_measures_dict[stock]['MAE_Vol'] = mae_vol
    
    t11 = time.time()
    print('time taken for one stock with ridge: ' ,datetime.timedelta(seconds = t11 - t01))
    print('*'*100)




# In[ ]:


measures_df = pd.DataFrame.from_dict(data = test_measures_dict, orient = 'index').reset_index()
test_pred_df = pd.DataFrame.from_dict(data = test_pred_dict, orient = 'index').reset_index()
measures_df.to_csv(res_file_name,index=None, header='column_names')
test_pred_df.to_csv(pred_file_name,index=None, header='column_names')


# In[ ]:



# # External Classifier

# In[18]:

def clfRF(Ztrain,Y_train,Ztest,Y_test,n_clf=5,depth=1,rnd_state=11):
 clf_rf = RandomForestClassifier(n_estimators=n_clf, max_depth=depth,random_state=rnd_state)
 clf_rf.fit(Ztrain, Y_train)
 ytr_rf_pred = clf_rf.predict(Ztrain)
 yte_rf_pred = clf_rf.predict(Ztest)
 tr_scores = clf_rf.predict_proba(Ztrain)
 te_scores = clf_rf.predict_proba(Ztest)
 return ytr_rf_pred, yte_rf_pred, tr_scores, te_scores



# In[19]:

rf_res_file_name = base_path+'Results2/Reg2/Classification/res_classification_measures.csv'
rf_pred_file_name = base_path+'Results2/Reg2/Classification/res_classification_pred.csv'
if os.path.exists(rf_res_file_name):
  os.remove(rf_res_file_name)
if os.path.exists(rf_pred_file_name):
  os.remove(rf_pred_file_name)
     
     
# In[22]:


cnt = 0 
pos_label = 1
depth = 3
num_clfs = 5
rf_test_measures_dict  = {}
final_results_df = pd.DataFrame()
t0 = time.time()
df_temp2 = pd.read_csv(base_path+'data/init.csv') 
for stock in stocks_list[start:end]:
 t01 = time.time()
 temp_dict = {}
 rf_test_measures_dict[stock] = {}
 _,windowed_data,_, _ = getWindowedDataReg(data_df,stock,window_size)
 feat_wise_data = getFeatWiseData(windowed_data,features_list)
 prev_day_values = getPrevDayFeatures(feat_wise_data)
 prev_day_values = prev_day_values[:,0]
 seed = int(df_temp2.loc[df_temp2['index']==stock]['seed'].values.tolist()[0])
 random_state = int(df_temp2.loc[df_temp2['index']==stock]['random_state'].values.tolist()[0])
 print('stock : ', stock)
 xtr_path = base_path + 'data/Reg2/TL_Train/' + stock + param_path +'_' + str(test_size) + '_tl_xtrain' + str(seed) + '.npy'
 ytr_path = base_path + 'data/Reg2/TL_Train/' + stock + param_path +  '_' + str(test_size) + '_tl_ytrain' + str(seed) + '.npy'
 xte_path = base_path + 'data/Reg2/TL_Test/' + stock + param_path +  '_' + str(test_size) + '_tl_xtest' + str(seed) + '.npy'
 yte_path = base_path + 'data/Reg2/TL_Test/' + stock + param_path +  '_' + str(test_size) + '_tl_ytest' + str(seed) + '.npy'
 Ztrain = np.load(xtr_path)
 Y_train = np.load(ytr_path)
 Ztest = np.load(xte_path)
 Y_test = np.load(yte_path)
 ytr_prev_day = prev_day_values[:Y_train.shape[0]]
 yte_prev_day  = prev_day_values[Y_train.shape[0]:]
 yte_prev_day  = yte_prev_day[:yte_prev_day.shape[0]-1]

 Y_train_true_labels = np.where((Y_train - ytr_prev_day)>0,1,0)
 Y_test_true_labels = np.where((Y_test - yte_prev_day)>0,1,0)
 ytr_pred, yte_pred, tr_scores, te_scores = clfRF(Ztrain,Y_train_true_labels, Ztest, Y_test_true_labels, n_clf=num_clfs,depth=depth, rnd_state=random_state)
 limit = Ztrain.shape[0]
 precision, recall, f1_score,_ = precision_recall_fscore_support(Y_test_true_labels, yte_pred, pos_label=1, average='binary')   
 print(f1_score)
 AR = compAnnualReturns(stock,yte_pred,data_df,window_size,limit)
 #print('AR: ',AR)
 fpr, tpr, thresholds = roc_curve(Y_test_true_labels, te_scores[:,pos_label], pos_label = pos_label)
 AUC_val = auc(fpr, tpr)
 rf_test_measures_dict[stock]['F1_score'] = round(f1_score,3)
 rf_test_measures_dict[stock]['Precision'] = round(precision,3)
 rf_test_measures_dict[stock]['Recall'] = round(recall,3)
 rf_test_measures_dict[stock]['AUC'] = round(AUC_val,3)
 rf_test_measures_dict[stock]['AR'] = AR
         
 temp_final_df = pd.DataFrame(Y_test,columns=['ytrue'])
 temp_final_df['ypred'] = yte_pred
 temp_scores_df = pd.DataFrame(te_scores) 
 temp_final_df = pd.concat([temp_final_df,temp_scores_df],axis = 1)
 temp_final_df['SYMBOL'] = stock
 final_results_df = pd.concat([final_results_df,temp_final_df],axis = 0)
 t11 = time.time()
 print('time taken for one stock with RF: ' ,datetime.timedelta(seconds = t11 - t01))
 print('*'*100)
t1 = time.time()
print('time taken for all stocks with RF: ' ,datetime.timedelta(seconds = t1 - t0))


measures_df = pd.DataFrame.from_dict(data = rf_test_measures_dict, orient = 'index').reset_index()
measures_df.to_csv(rf_res_file_name,index=None, header='column_names')
final_results_df.to_csv(rf_pred_file_name,index=None, header='column_names')





