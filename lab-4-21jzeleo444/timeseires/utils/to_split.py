# -*- coding: utf-8 -*-
"""
Created on 3:27 AM Tuesday, 6 September 2022

@author: Syed Hasnat
"""
import numpy as np
def to_split(dataset, time_steps, target_col, target_len):
    '''
    \nto_split(dataset, time_steps,target_col,target_len)\n
    dataset=nd.array
    time_steps= look back the previous data points like 24hrs, 168 hours etc 
    target_col= index of the column to which you want to pridect, it may be a single column or a list of columns
    target_len= how many point do you want to predict
    \n\nExample:
    to call this function
    look_back = 14*24
    train_X,train_y=to_split(train_set, time_steps= look_back,target_col=0,target_len=1)               #target_col=[0,1]
    validation_X,validation_y=to_split(validation_set, time_steps=look_back,target_col=0,target_len=1) #target_col=[0,1]
    test_X,test_y=to_split(test_set,time_steps=look_back,target_col=0,target_len=1)                    #target_col=[0,1]
    '''
    X,y = list(), list()
    for i in range(len(dataset)):
        end_of_x = i + time_steps
        if end_of_x > len(dataset)-1:
            break
        seq_x = dataset[i:end_of_x, :dataset.shape[1]]
        X.append(seq_x)
        
        target_list = list()
        for i in range(target_len):
            target_list.append(i+1)
       
        for x in target_list:
            another_end=end_of_x+x
            seq_y=dataset[end_of_x:another_end, target_col]
            if another_end==end_of_x+target_len:    
                y.append(seq_y)
    return np.array(X), np.array(y)