import numpy as np
# -*- coding: utf-8 -*-
"""
Created on 3:27 AM Tuesday, 6 September 2022

@author: Syed Hasnat
"""
def t_v_t(dataset,train_p,validation_p,test_p):
    '''
    \n\nt_v_t(dataset,train_p,validation_p,test_p)\n
    dataset = DataFrame
    train_p = Percentage of Traing Data
    validation_p = Percentage of Validation Data
    test_p = Percentage of Test Data
    \n\nExample:
    to call this Function;
    train_set , validation_set , test_set = t_v_t(df,70,20,10)
    '''
    train_p=train_p/100
    validation_p=validation_p/100
    test_p=test_p/100
    train=int(np.round(len(dataset)*train_p))                                 #70
    validation=int(np.round(len(dataset)*validation_p))                       #20
    test=int(np.round(len(dataset)*test_p))                                   #10
    train_set,validation_set,test_set=dataset[:train],dataset[train:train+validation],dataset[train+validation:len(dataset)]
    return train_set,validation_set,test_set