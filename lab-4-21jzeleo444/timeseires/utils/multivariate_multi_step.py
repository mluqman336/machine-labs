# -*- coding: utf-8 -*-
"""
Created on 3:27 AM Tuesday, 6 September 2022

@author: Syed Hasnat
"""
import numpy as np
def multivariate_multi_step(dataset, time_steps, target_col,target_len,ahead):
    '''
    \numultivariate_multi_step(dataset, time_steps,target_col,target_len)\n
    dataset=nd.array
    time_steps= look back the previous data points like 24hrs, 168 hours etc 
    target_col= index of the column to which you want to pridect,
    target_len= how many point do you want to predict
    ahead= how many point do you want to skip
    \n\nExample:
    to call this function
    look_back = 14*24
    train_X,train_y=multivariate_multi_step(train_set, time_steps= look_back,target_col=0,target_len=2,ahead=4)               
    validation_X,validation_y=multivariate_multi_step(validation_set, time_steps=look_back,target_col=0,target_len=2,ahead=4) 
    test_X,test_y=multivariate_multi_step(test_set,time_steps=look_back,target_col=0,target_len=2,ahead=4)                    
    '''
    X,y = list(), list()
    for i in range(len(dataset)):
        end_of_x = i + time_steps
        if (end_of_x+target_len+ahead)> len(dataset)-1:
            break
        seq_x = dataset[i:end_of_x, :dataset.shape[1]]
        X.append(seq_x)
        target_column = dataset[:,target_col]
        seq_y=target_column[end_of_x+ahead:end_of_x+target_len+ahead]    
        y.append(seq_y)
    return np.array(X), np.array(y) #.reshape(-1,1)