import finalfunctions as ff
import neuralnet as nn
import numpy as np
import pandas as pd
import csv
import random


loop = ff.Final()  # calls class epoch cycle

flag = False
accuratepredictions = 0
#trainingset = range(0,700)
#testingset = range (700,1000)

np.random.seed (0)
w = np.random.uniform(size=(10,784),low = 0, high= 1) 
np.random.seed(0)
b = np.random.uniform(size=(1,10),low = 0, high= 1)  # very small biases

############################################################################################
np.random.seed (1)
w1 = np.random.uniform(size=(10,10),low = 0, high= 1) 
np.random.seed(1)
b1 = np.random.uniform(size=(1,10),low = 0, high= 1)  # very small biases

#############################################################################################

dw1,dw,db1,db = 0,0,0,0



for k in range(0,5): #trainingset:  # loops through images. 90 sec = 10 images image 0 and forward 

    for i in range (1): # iterations + 1(this is the base)  = 2 cycles back and forward per image
        
        indloss,it,wl0,bl0,wl1,bl1,wdeltal0,bdeltal0,wdeltal1,bdeltal1 = loop.inductiveloop(flag,k,w,b,w1,b1,dw,db,dw1,db1) # a = b            
        flag,k,w,b,w1,b1,dw,db,dw1,db1 = indloss,it,wl0,bl0,wl1,bl1,wdeltal0,bdeltal0,wdeltal1,bdeltal1
        #print (F"WEIGHTS l0 TO L1 : {wdeltal0}\n BIASES l0 TO L1 : {bdeltal0}\n ")
        #print (F"WEIGHTS l1 TO L2 : {wdeltal1}\n BIASES l1 TO L2 : {bdeltal1}\n ")

        if indloss == True:
            print (f'INDLOSS {indloss}, {k}')
            accuratepredictions+=1
            break       

accuracy = accuratepredictions / 5 # len (trainingset) # change to len (testingset) when running testing set.

print (f'\n ACCURACY  ::: {accuracy * 100}%')

########################################################################################################################
