import neuralnet as nn 
import numpy as np
import pandas as pd
import csv
import random
import sys

class Final():


    def inductiveloop(self,flagbool,kth,w,b,w1,b1):   #,dw,db,dw1,db1): # add a k for iterations.

        #print (w, "first w ")
        #print (b, "first  B")
        #print (w1, "first  W1")
        #print (b1, "first B1")
        
        #print (dw[0], "first DW")
        #print (db, "first DB")
        #print (dw1, "first DW1")
        #print (db1, "first DB1")


        # THESE ARE 0 in the first iteration dw,db,dw1,db1

        ######### initalizing data  #################
        Fullcycle= nn.EpochCycle()  # calls class epoch cycle
        realarr = Fullcycle.replacewithrealarray(kth) #(3)# read all 42000. For loop. Gets data from index 4 
        
        correctlabel = Fullcycle.accuratelabel()
        
        ########################### Starting Forward Prop ###########################

        l0tol1nect = Fullcycle.inductivedotproductl0tol1(w,b,realarr)   #position  = Fullcycle.RELU (randomarr) # for position purposes
          
        #ogweightsl0tol1 = Fullcycle.originalweightsl0tol1() # ? Perhaps not needed
        #ogbiasesl0tol1 = Fullcycle.originalbiasesl0tol1() # ? Perhaps not needed

        reludone = Fullcycle.RELU(l0tol1nect)  # After Relu. zero to 1 connections complete with act.
        derivrelu = Fullcycle.PDRELU(reludone) 

        l1tol2 = Fullcycle.inductivedotproductl1tol2(w1,b1) # !changed

        sftmax = Fullcycle.Softmax(l1tol2) # After Softmax. 1 to 2 connections complete with activation.
        ########################## Finishing Forward Prop ############################################

        badimg = Fullcycle.explodedgradientresolved() # ? Perhaps not needed
    

        ######## going backwards and updating dw1, db1, dw, db ##########


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
        
        chainsoftcross = Fullcycle.CEntropywithsoftchainrule(crossderive,sftgrad) # Cross entropy & Softmax Chain Rule derivative
        
        nw1 =  Fullcycle.newweightl2tol1(chainsoftcross,sftgrad, reludone, kth) # New weights l2 to l1
        nb1 =  Fullcycle.newbiasl2tol1(chainsoftcross,sftgrad, kth) # New bias l2 to l1

        nw =  Fullcycle.newweightl1tol0(chainsoftcross,derivrelu,realarr,w1, kth)  # New weights l1 to l0 
        nb =  Fullcycle.newbiasl1tol0(chainsoftcross,derivrelu, kth)   # New bises l1 to l0 
        #print (nb , "New bias")
    
        
        ########################## Finishing Backward Prop ############################################
            
        #print (nb)
        #print (b)


        w,b,w1,b1= Fullcycle.updatewandb(w,b,w1,b1,nw,nb,nw1,nb1,0.01)

        return flagbool,kth,w,b,w1,b1  #,dw,db,dw1,db1


