import neuralnet as nn 
import numpy as np
import pandas as pd
import csv
import random
import sys

class GENERATE():


    def pixelsum(self,it,drawnum,w,b,w1,b1,dw,db,dw1,db1):
        ######### initalizing data  #################
        Fullcycle= nn.EpochCycle()  # calls class epoch cycle
  
        randomarr = Fullcycle.replacerowwithrand()#(3)# read all 42000. For gen. Gets data from index 4 
        randomsum = Fullcycle.randomsum(randomarr)
        
        if it != 0:
            w1 = Fullcycle.updatelearnl2tol1weights(dw1,w1,0.9)#1.540009)# 1.0011) # updatel2tol1we =  updataed - old * lr
            b1 = Fullcycle.updatelearnl2tol1biases(db1,b1,0.9)#1.540009)#1.0011)
            
            w = Fullcycle.updatelearnl1tol0weights(dw,w,0.9)#) #1.0011) # cant be 0.001 or 0.01 
            b = Fullcycle.updatelearnl1tol0biases(db,b,0.9)#1.540009) #1.0011)
    

        print (f'\niteration:: {it}')
        
        connectsl0 = Fullcycle.inductivedotproductl0tol1(w,b,randomarr)   #position  = Fullcycle.RELU (randomarr) # for position purposes
    
        targetsum = Fullcycle.averageofsupersum(drawnum)     
        targetimage = Fullcycle.imagetargeted(drawnum)
        
        posoftarget = Fullcycle.PDRELU (targetimage)
        
        apprelu = Fullcycle.RELU(connectsl0) 

        maxrelu = Fullcycle.maxRELUvalue()
        posofbestrelu = Fullcycle.positionofmaxRELUvalue(apprelu,targetsum)

        wtbt = Fullcycle.maxreluafterdot(w,b,posofbestrelu) # for l0tol1 weights
        

        imagegenerated = Fullcycle.newimage(wtbt,randomarr)
        mapingvalues = Fullcycle.mappingwithvalues(imagegenerated,posoftarget)

        tloss = Fullcycle.imageabsoluteloss(targetsum,mapingvalues) # this is its own thing

        printnew = Fullcycle.writeimage(mapingvalues)

        print (f"{targetsum} - sum of generated image =  ::: {tloss}")

        derivofrelu = Fullcycle.PDRELU (connectsl0)

        connectsl1 = Fullcycle.inductivedotproductl1tol2(w1,b1)
        softmaxwithconnects  = Fullcycle.Softmax(connectsl1) # could be reshaped. 
        softmaxpartials = Fullcycle.Softmaxpartialderivatives(softmaxwithconnects)#softmax partial derivatives to go back. 

        derivmae = Fullcycle.derivativeMAE(apprelu,targetsum)

        firstchain = Fullcycle.CEntropywithsoftchainrule(derivmae,softmaxpartials) # change the name of this name. 
        secondchain = Fullcycle.inductivenextchainrule(w1,firstchain,derivofrelu)

        dw1 = Fullcycle.newweightl2tol1(firstchain,apprelu)
        db1 = Fullcycle.newbiasl2tol1(firstchain)

        dw = Fullcycle.newweightl1tol0(secondchain,randomarr)
        db = Fullcycle.newbiasl1tol0(secondchain)

        #print (f'relu values :::{apprelu}')
        #print (f'ROC {maxrelu}')
        #print (f'targetsum - maxrelu ( {targetsum} - {maxrelu} )  = {targetsum - maxrelu}')
        
        boolsettotrue = False

        if -0.85 * targetsum < tloss < 0.85 * targetsum:  # 85 percent accuracy
            boolsettotrue = True

        # relu array and max relu to yeild position,
        return posofbestrelu,boolsettotrue,it,w,b,w1,b1,dw,db,dw1,db1


