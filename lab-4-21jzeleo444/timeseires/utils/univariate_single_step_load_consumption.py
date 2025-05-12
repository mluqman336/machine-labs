# -*- coding: utf-8 -*-
"""
Created on 3:27 AM Tuesday, 6 September 2022

@author: Syed Hasnat
"""
import numpy as np
import numpy as np
def univariate_single_step_load_consumption(dataset, time_steps, target_col):
    '''
    \nunivariate_single_step_load_consumption(dataset, time_steps,target_col,target_len)\n
    dataset=nd.array
    time_steps= look back the previous data points like 24hrs, 168 hours etc 
    target_col= index of the column to which you want to pridect, it may be a single column or a list of columns
    \n\nExample:
    to call this function
    look_back = 14*24
    train_X,train_y=univariate_single_step_load_consumption(train_set, time_steps= look_back,target_col=0)    
    validation_X,validation_y=univariate_single_step_load_consumption(validation_set,time_steps=look_back,target_col=0) 
    test_X,test_y=univariate_single_step_load_consumption(test_set,time_steps=look_back,target_col=0)         
    '''
    X,y = list(), list()
    for i in range(len(dataset)):
        end_of_x = i + time_steps
        if end_of_x > len(dataset)-1:
            break
        seq_x = dataset[i:end_of_x, :1]
        X.append(seq_x)
        #target_column = dataset[:,0:1] for future ka chari dwa columns forecast kol wo
        #seq_y=target_column[end_of_x]
        seq_y=dataset[end_of_x, 0]#target_col]    
        y.append(seq_y)
    return np.array(X), np.array(y).reshape(-1,1) #.reshape(-1,1)