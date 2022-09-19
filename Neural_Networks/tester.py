import finalfunctions as ff
import neuralnet as nn
import numpy as np
import pandas as pd
import csv
import random


loop = ff.Final()  # calls class epoch cycle

# change weights and biases and everything else. 


np.random.seed (0)  # size(sets, nodes)
w = np.random.uniform(size=(533,784),low = -1, high= 1) * np.sqrt(2 / 784) #maybe change to -1 as low 
np.random.seed(0)
b = np.random.uniform(size=(533,1),low = 0, high= 1) * 0   # very small biases #b = np.random.uniform(size=(533,1),low = -1, high= 1) * np.sqrt(2 / 533) 

############################################################################################
np.random.seed (1)
w1 = np.random.uniform(size=(533,10),low = -1, high= 1) * np.sqrt(2 / 533)#maybe change to -1 as low 
np.random.seed(1)
b1 = np.random.uniform(size=(10,1),low = 0, high= 1) * 0  # very small biases # b1 = np.random.uniform(size=(10,1),low = -1, high= 1) *  np.sqrt(2 / 10) 

#############################################################################################

dw = np.zeros((533,784))
db = np.zeros((533,1))

dw1 = np.zeros((533,10))
db1 = np.zeros((10,1))
#dw1,db1,dw,db = 0,0,0,0


flag = False
accuratepredictions = 0

upto = 1000 #29400  # 1,0,1,4,0,0



for k in range(0,upto): #trainingset:  # loops through images. 90 sec = 10 images image 0 and forward 

    print (k)
    tf,it,w0,b0,wl1,bl1 = loop.inductiveloop(flag,k,w,b,w1,b1) # a = b            
    
    flag,k,w,b,w1,b1 = tf,it,w0,b0,wl1,bl1

        

    #print (f"\ni:{k} b::::::::::::::::::::::;;    {nb} ")
    #print (F"WEIGHTS l0 TO L1 : {wdeltal0}\n BIASES l0 TO L1 : {bdeltal0}\n ")
    #print (F"WEIGHTS l1 TO L2 : {wdeltal1}\n BIASES l1 TO L2 : {bdeltal1}\n ")

    if tf == True:
        print (f'tf {tf}, {k}')
        accuratepredictions+=1
        #break       

accuracy = accuratepredictions / upto # len (trainingset) # change to len (testingset) when running testing set.

print (f'\n ACCURACY  ::: {accuracy * 100}%')

########################################################################################################################
