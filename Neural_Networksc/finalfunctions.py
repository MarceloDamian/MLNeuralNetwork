import neuralnet as nn 
import numpy as np
import pandas as pd
import csv
import random
import sys

class Final():


    def inductiveloop(self,flagbool,kth,w,b,w1,b1,dw,db,dw1,db1): # add a k for iterations.

        ######### initalizing data  #################
        Fullcycle= nn.EpochCycle()  # calls class epoch cycle
        realarr = Fullcycle.replacewithrealarray(kth) #(3)# read all 42000. For loop. Gets data from index 4 
        
        correctlabel = Fullcycle.accuratelabel()
        
        ########################### Starting Forward Prop ###########################

        w1 = Fullcycle.updatelearnl2tol1weights(dw1,w1,0.7)#1.540009)# 1.0011) # updatel2tol1we =  updataed - old * lr
        b1 = Fullcycle.updatelearnl2tol1biases(db1,b1,0.7)#1.540009)#1.0011)
        
        w = Fullcycle.updatelearnl1tol0weights(dw,w,0.7)#) #1.0011) # cant be 0.001 or 0.01 
        b = Fullcycle.updatelearnl1tol0biases(db,b,0.7)#1.540009) #1.0011)

        l0tol1nect = Fullcycle.inductivedotproductl0tol1(w,b,realarr)   #position  = Fullcycle.RELU (randomarr) # for position purposes
          
        #ogweightsl0tol1 = Fullcycle.originalweightsl0tol1() # ? Perhaps not needed
        #ogbiasesl0tol1 = Fullcycle.originalbiasesl0tol1() # ? Perhaps not needed

        reludone = Fullcycle.RELU(l0tol1nect)  # After Relu. zero to 1 connections complete with act.
        derivrelu = Fullcycle.PDRELU(reludone) 

        l1tol2 = Fullcycle.inductivedotproductl1tol2(w1,b1) # !changed

        sftmax = Fullcycle.Softmax(l1tol2) # After Softmax. 1 to 2 connections complete with activation.
        ########################## Finishing Forward Prop ############################################

        badimg = Fullcycle.explodedgradientresolved() # ? Perhaps not needed
    
        ######################### Starting Backward Prop #########################
        sftgrad = Fullcycle.Softmaxpartialderivatives(sftmax) # Softmax partial derivatives or gradients
        sfthotencode = Fullcycle.SoftmaxHotencode(sftmax) 
           
        crossentr = Fullcycle.Cross_entropy(correctlabel,sftmax) # array,labeltarget # Cross entropy on Softmax
        
        lowcost=Fullcycle.maxcrossentropy() # NEEDED FOR MAX
        print (f'CROSS ENTROPY::::: {Fullcycle.maxcrossentropy()}')

        actispred,Y,Ypred,correctpred = Fullcycle.Probability(correctlabel)
        print (f'\nCORRECT LABEL::: {Y}  PREDICTED LABEL:::  {Ypred}  Probability Of Correct Label :::  {correctpred}\n')

        if actispred==True :
            print ('   BASE::::  SYS.EXIT: ACCURATE == PREDICTED   TRUE     \n')
                
        flagbool = actispred





        crossderive = Fullcycle.CEntropyderivative(correctlabel,sftmax)  # array,labeltarget # Cross entropy derivative 
        
        chainsoftcross = Fullcycle.CEntropywithsoftchainrule(crossderive,sftgrad) # Cross entropy & Softmax Chain Rule derivative
        nxtchainrule =  Fullcycle.inductivenextchainrule(w1,chainsoftcross,derivrelu) #  Chain rule on Relu deriv and weights and losserror

        dw1 =  Fullcycle.newweightl2tol1(chainsoftcross,reludone) # New weights l2 to l1
        db1 =  Fullcycle.newbiasl2tol1(chainsoftcross) # New bias l2 to l1

        dw =  Fullcycle.newweightl1tol0(nxtchainrule,realarr)  # New weights l1 to l0 
        db =  Fullcycle.newbiasl1tol0(nxtchainrule)   # New bises l1 to l0 
        ########################## Finishing Backward Prop ############################################

        return flagbool,kth,w,b,w1,b1,dw,db,dw1,db1


