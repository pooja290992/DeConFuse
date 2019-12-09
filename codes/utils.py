import matplotlib.pyplot as plt
from sklearn.metrics import *
import seaborn as sb
import pandas as pd
import numpy as np 
from data_processing import *

res_path = '../Results/'
def computeConfMatrix(ytrue,ypred,title):
    cm = confusion_matrix(ytrue, ypred)
    tn, fp, fn, tp = confusion_matrix(ytrue,ypred).ravel()
    sb.set(font_scale=1.4)
    labels = [0,1]
    cm_df = pd.DataFrame(cm,columns= labels,index=labels)
    sb.heatmap(cm_df, annot=True,annot_kws={"size": 16},fmt='d')
    plt.ylabel('Actual labels')
    plt.xlabel('Predicted labels') 
    plt.title('Confusion Matrix for ' + title)
    plt.show()
    return cm,tn, fp, fn, tp
    


def plotROC(ytrue,scores,title,pos_class = 1):    
    fpr, tpr, _ = roc_curve(ytrue, scores[:,pos_class], pos_label=pos_class)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr,label=str(pos_class) + ' as +ve class(auc = %0.2f)' % roc_auc)
    lw = 2
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for '+title)
    legend = plt.legend(loc="lower right",frameon=True)
    legend.get_frame().set_edgecolor('black')
    plt.show()
    return fpr, tpr



def plotPrecisionRecall(ytrue,scores,title,pos_class = 1):    
    precision, recall, _ = precision_recall_curve(ytrue, scores[:,pos_class], pos_label=pos_class)
    average_precision = average_precision_score(ytrue, scores[:,pos_class])
    plt.plot(recall, precision,label=str(pos_class) + ' as +ve class(ap = %0.2f)' % average_precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for '+title)
    legend = plt.legend(loc="upper right",frameon=True)
    legend.get_frame().set_edgecolor('black')
    plt.show()
    return precision, recall 

    
    
def computeMetrics(tp,tn,fp,fn):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    sensitivity = recall
    specificity = tn / (tn + fp)
    tpr = recall
    fpr = fp / (fp + tn)
    f1score = 2*precision*recall/(precision+recall)
    return precision, recall, sensitivity, specificity, tpr,fpr, f1score



def getStocksList(data_df):
    g = data_df.groupby('SYMBOL')
    df1 = g.count().reset_index()
    df1 = df1[['SYMBOL','CLOSE']].reset_index()
    df1.rename(columns= {'CLOSE':0},inplace=True)
    stocks_list = df1.loc[df1[0]>807]['SYMBOL'].values.tolist()
    print('length : ' , len(stocks_list))
    return stocks_list



def compAnnualReturns(stock,ypred,data_df,window_size,limit,sub_one=True):
    _,_,stock_table, _ = getWindowedDataReg(data_df,stock,window_size)
    stock_table_df = pd.DataFrame(stock_table,columns = ['CLOSE','OPEN','HIGH','LOW','CONTRACTS','DATE','Stock_Class'])
    stock_table_df  = stock_table_df[limit:]
    if sub_one:
        stock_table_df  = stock_table_df[0:stock_table_df.shape[0]-1]
    stock_table_df['Predicted'] = ypred

    i = 0
    startCapital = 100000.0
    totalTransactionLength = 0
    buyPoint = 0
    sellPoint= 0
    gain = 0.0
    totalGain = 0.0
    money = startCapital
    shareNumber = 0.0
    moneyTemp = 0.0
    maximumMoney = 0.0 
    minimumMoney = startCapital
    maximumGain = 0.0
    maximumLost = 100.0
    totalPercentProfit = 0.0
    transactionCount = 0
    successTransactionCount = 0
    failedTransactionCount = 0
    buyPointBAH = 0
    shareNumberBAH = 0
    moneyBAH = startCapital
    maximumProfitPercent = 0.0
    maximumLostPercent = 0.0
    forceSell = False
    transactionCharges = 10
    rows,cols = stock_table_df.shape
    k = 0
    #print(rows,cols)
    num_days = 0 
    while(k<rows):
        if(stock_table_df.iloc[k]['Predicted'] == 0):
            buyPoint = stock_table_df.iloc[k]['CLOSE']
            buyPoint = buyPoint*100
            shareNumber = (money-transactionCharges)/buyPoint
            forceSell = False
    
            for j in range(k,rows):
                sellPoint = stock_table_df.iloc[j]['CLOSE']
                sellPoint = sellPoint*100;
                moneyTemp = (shareNumber*sellPoint)-transactionCharges
                    

                if(stock_table_df.iloc[j]['Predicted'] == 1 or forceSell == True):
                    sellPoint = stock_table_df.iloc[j]['CLOSE']
                    sellPoint = sellPoint*100
                    
                    
                    gain = sellPoint-buyPoint
              
                    if(gain>0):
                        successTransactionCount += 1

                    else:
                        failedTransactionCount += 1


                    if(gain >= maximumGain):
                        maximumGain = gain
                        maximumProfitPercent = (maximumGain/buyPoint)*100;		

                    if(gain <= maximumLost):
                        maximumLost = gain
                        maximumLostPercent = (maximumLost/buyPoint)*100		

                    moneyTemp = (shareNumber*sellPoint)-transactionCharges
                    money = moneyTemp

                    if(money > maximumMoney):
                        maximumMoney = money

                    if(money < minimumMoney):
                        minimumMoney = money

                    transactionCount += 1

                    #print(str(transactionCount) + "." + "("+str(k+1)+"-"+str(j+1)+") => " + str(round((gain*shareNumber),2)) + " Capital: Rs" + str(round(money,2)))

                    totalPercentProfit = totalPercentProfit + (gain/buyPoint);

                    totalTransactionLength = totalTransactionLength + (j-k);
                    k = j+1
                    totalGain = totalGain + gain
                    break
        k += 1

    startDate = datetime.datetime.strptime(stock_table_df.iloc[0]['DATE'], "%Y-%m-%d").date()
    endDate = datetime.datetime.strptime(stock_table_df.iloc[rows-1]['DATE'], "%Y-%m-%d").date()

    dt = endDate-startDate
    print('days:',dt.days)
    numberOfDays = dt.days
    numberOfYears = numberOfDays/365
    if transactionCount == 0:
        transactionCount = 1e-5
    AR = round(((math.exp(math.log(money/startCapital)/numberOfYears)-1)*100),2)
    return AR
