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
        realarr = Fullcycle.replacewithrealarray(kth) #(3)# read all 42000. For loop. Gets data from index 4 
        
        correctlabel = Fullcycle.accuratelabel()
        
        ########################### Starting Forward Prop ###########################

        l0tol1nect = Fullcycle.inductivedotproductl0tol1(w,b,realarr)   #position  = Fullcycle.RELU (randomarr) # for position purposes

        reludone = Fullcycle.RELU(l0tol1nect)  # After Relu. zero to 1 connections complete with act.
        derivrelu = Fullcycle.PDRELU(reludone) 

        l1tol2 = Fullcycle.inductivedotproductl1tol2(w1,b1) # !changed

        sftmax = Fullcycle.Softmax(l1tol2) # After Softmax. 1 to 2 connections complete with activation.
        ########################## Finishing Forward Prop ############################################
        ######## going backwards and updating  ##########


        ######################### Starting Backward Prop #########################
        sftgrad = Fullcycle.Softmaxpartialderivatives(sftmax) # Softmax partial derivatives or gradients
        sfthotencode = Fullcycle.SoftmaxHotencode(sftmax) 
           
        crossentr = Fullcycle.Cross_entropy(correctlabel,sftmax) # array,labeltarget # Cross entropy on Softmax

        lowcost=Fullcycle.maxcrossentropy() # NEEDED FOR MAX
        print (f'CROSS ENTROPY::::: {Fullcycle.maxcrossentropy()}')

        actispred,Y,Ypred,correctpred = Fullcycle.Probability(correctlabel)
        #!
        print (f'\nCORRECT LABEL::: {Y}  PREDICTED LABEL:::  {Ypred}  Probability Of Correct Label :::  {correctpred * 100} %\n')

        if actispred==True :
            print ('   BASE::::  SYS.EXIT: ACCURATE == PREDICTED   TRUE     \n')
                
        flagbool = actispred

        crossderive = Fullcycle.CEntropyderivative(correctlabel,sftmax)  # array,labeltarget # Cross entropy derivative 
        ccloss = Fullcycle.CEntropywithsoftchainrule(crossderive,sftgrad) # Cross entropy & Softmax Chain Rule derivative

        nw,nb,nw1,nb1 = Fullcycle.errorbackprop(sftgrad,realarr,reludone,ccloss,derivrelu,crossderive,w1,w,b1,kth)

        ########################## Updating weights ... ############################################
        
        w,b,w1,b1= Fullcycle.updatewandb(w,b,w1,b1,nw,nb,nw1,nb1,0.1)

        # 0.01 goes down 
        # 0.1 yeilds 26 percent for 100. 31.9 percent for 1000 images
        # 0.2 yields 31 percent for 100 and 61 percent for 29,400 
        # 0.22 yeilds 32 percent for 100 
        # 0.23 yeilds 33 percent for 100         

        #EDIT: FOR 0.2 IT YEILDS 40 percent for 1000 images

        return flagbool,kth,w,b,w1,b1 


