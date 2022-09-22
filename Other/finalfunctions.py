import neuralnet as nn 
import numpy as np
import pandas as pd
import csv
import random
import sys

class Final():


    def inductiveloop(self,flagbool,kth,w,b,w1,b1):   #,dw,db,dw1,db1): # add a k for iterations.

        ######### initalizing data  #################
        Fullcycle= nn.EpochCycle()  # calls class epoch cycle
        
        Fullcycle.replacewithrealarray(kth) #(3)# read all 42000. For loop. Gets data from index 4         
        Fullcycle.accuratelabel()
        
        ########################### Starting Forward Prop ###########################
        Fullcycle.inductivedotproductl0tol1(w,b)   #position  = Fullcycle.RELU (randomarr) # for position purposes
        Fullcycle.RELU()  # After Relu. zero to 1 connections complete with act.
        Fullcycle.PDRELU() 
        Fullcycle.inductivedotproductl1tol2(w1,b1) # !changed
        Fullcycle.Softmax() # After Softmax. 1 to 2 connections complete with activation.

        ######################### Starting Backward Prop #########################
        Fullcycle.Softmaxpartialderivatives() # Softmax partial derivatives or gradients
        Fullcycle.SoftmaxHotencode() 
        Fullcycle.Cross_entropy() # array,labeltarget # Cross entropy on Softmax
        
        Fullcycle.maxcrossentropy() # NEEDED TO PRINT
        
        actispred= Fullcycle.Probability()
        

        Fullcycle.CEntropyderivative()  # array,labeltarget # Cross entropy derivative 
        Fullcycle.CEntropywithsoftchainrule() # Cross entropy & Softmax Chain Rule derivative
       
        nw,nb,nw1,nb1 = Fullcycle.errorbackprop(w1,w,b1,kth)

        ########################## Updating weights ... ############################################
        w,b,w1,b1= Fullcycle.updatewandb(w,b,w1,b1,nw,nb,nw1,nb1,kth,0.9, 0.23)

        # 0.01 goes down 
        # 0.1 yeilds 26 percent for 100. 31.9 percent for 1000 images
        # 0.2 yields 31 percent for 100. 39.9% for 1000 images. 61 percent for 29,400.
        # 0.22 yeilds 32 percent for 100. 41.4 percent for 1000 images
        # 0.23 yeilds 33 percent for 100.  42.9 percent for 1000 images    
        # 0.24 yeilds 32 percent for 100.  yeilds 43.3 percent for 1000
        # 0.25 yeilds 42.9 percent for 1000.
        #EDIT: FOR 0.2 IT YEILDS 40 percent for 1000 images

        return actispred,kth,w,b,w1,b1 


