import finalfunctions as ff
import neuralnet as nn
import numpy as np
import pandas as pd
import csv
import random


loop = ff.Final()  # calls class epoch cycle


np.random.seed (0)
w = np.random.uniform(size=(10,784),low = 0, high= 1) #maybe change to -1 as low 
np.random.seed(0)
b = np.random.uniform(size=(1,10),low = 0, high= 1)[0]  # very small biases

############################################################################################
np.random.seed (1)
w1 = np.random.uniform(size=(10,10),low = 0, high= 1) #maybe change to -1 as low 
np.random.seed(1)
b1 = np.random.uniform(size=(1,10),low = 0, high= 1)[0]  # very small biases

#############################################################################################

dw = np.zeros((10,784))
db = np.zeros((1,10))[0]

dw1 = np.zeros((10,10))
db1 = np.zeros((1,10))[0]
#dw1,db1,dw,db = 0,0,0,0


flag = False
accuratepredictions = 0

for k in range(0,2): #trainingset:  # loops through images. 90 sec = 10 images image 0 and forward 

    #for i in range (1): # iterations + 1(this is the base)  = 2 cycles back and forward per image

    tf,it,  w0,b0,wl1,bl1,nw,nb,nw1,nb1 = loop.inductiveloop(flag,k,   w,b,w1,b1,dw,db,dw1,db1) # a = b            
    #flag,k,w,b,w1,b1,dw,db,dw1,db1 = tf,it,nw,nb,nw1,nb1,w0,b0,wl1,bl1
    #tf,it,w0,b0,wl1,bl1,nw,nb,nw1,nb1 = flag,k,w,b,w1,b1,dw,db,dw1,db1 =
    #nw1 is presenting itself as negative 0's. 

    print (f"\ni:{k} bl1::::::::::::::::::::::;;    {w0} ")


    #print (f"\nw::::::::::::::::::::::;;    {w}")
    #print (f"\nnw::::::::::::::::::::::    {nw}")
    #tf,it,nw,nb,nw1,nb1,w0,b0,wl1,bl1 = flag,k,w,b,w1,b1,dw,db,dw1,db1
    

#    flag,k,w,b,w1,b1,dw,db,dw1,db1 = tf,it,nw,nb,nw1,nb1,w0,b0,wl1,bl1


    #print (F"WEIGHTS l0 TO L1 : {wdeltal0}\n BIASES l0 TO L1 : {bdeltal0}\n ")
    #print (F"WEIGHTS l1 TO L2 : {wdeltal1}\n BIASES l1 TO L2 : {bdeltal1}\n ")

    if tf == True:
        print (f'tf {tf}, {k}')
        accuratepredictions+=1
        #break       

accuracy = accuratepredictions /2 # len (trainingset) # change to len (testingset) when running testing set.

print (f'\n ACCURACY  ::: {accuracy * 100}%')

########################################################################################################################
