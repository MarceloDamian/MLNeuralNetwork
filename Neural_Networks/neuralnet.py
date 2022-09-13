import numpy as np
import pandas as pd
import csv
import random
import copy
from scipy.special import expit


class EpochCycle():

    def replacewithrealarray(self, rnumber): # replaces with one picture from real array 
                
        numpyarray = np.array([])

        with open('./train.csv', 'r') as csv_file: # probably changed to train.csv
            csvreader = csv.reader(csv_file) # loads data into csv reader
            next(csvreader) # skips pixelnumbers skips 0 
            for index, row in enumerate(csvreader): # enumerates file into indexes
               if index==rnumber:  # Load image at row given
                    pixels = row[1:] # for train.csv ignores or skips over the 0th column  
                    self.label = row[0] #! These are the labels of train.csv 
            
            numpyarray = np.append (numpyarray, list(map(float, pixels)) ) # map characthers as integers

        return numpyarray/255 # full image is established in nested list. Divided by 255 to get value under 1.
    
    def accuratelabel(self):
        return self.label
    
    def gentargetlabel(self,target):
        return target
    
    def replacerowwithrand(self):
        self.insidenest = np.random.uniform(size=(1,784),low = 0, high= 256) 
        np.random.seed(0)
        return self.insidenest[0]

    def randomsum(self, randomarray):
        return np.sum (randomarray)


    #! FIX UDPDATED LEARNING ALL OF THEM.  REVIEW GRADIENT DESCENT. 

    def updatewandb(self,w,b,w1,b1,dw,db,dw1,db1,learnrate):
    

        newprimel0w = w - (dw * learnrate) 
        newprimel0b = b - (db * learnrate)
        
        newprimel1w = w1 - (dw1 * learnrate) 


        newprimel1b = b1- (db1 * learnrate)

        #print ("\nthis one ",newprimel1w, "i=1  ")

        # b is lessrandom. and 

        #print (b, "first  B")
        #print (w1, "first  W1")
        #print (b1, "first B1")
        
        #print (dw, "first DW")
        #print (db, "first DB")
        #print (dw1, "first DW1")
        #print (db1, "first DB1")

            
        return newprimel0w,newprimel0b,newprimel1w,newprimel1b

    def inductivedotproductl0tol1 (self, inductweights, inductbiases, arrayofpixels):
        
        npinsertarr = arrayofpixels.reshape(784,1) # reduced this was next line : npinsertarr = npinsertarr.reshape(784,1)
        
        #! change mutliplication to dot because it isnt supposed to just get multiplied 
        weightwithbias = np.dot (inductweights, npinsertarr) + inductbiases.reshape (10,1) # maybe inductweights @ npinsertarr instead of dot. 
        
        inductivebeforeRelu = weightwithbias.reshape(10) #.tolist()

        return inductivebeforeRelu

    def originalweightsl0tol1 (self):
        return self.saveweightsl0tol1

    def originalbiasesl0tol1 (self):
        return self.savebiasesl0to1

    def averageofsupersum (self, neuron):

        with open('./sumdp.csv', 'r') as csv_file:
            csvreader = csv.reader(csv_file)

            for data in csvreader:
                #print (f'data [0]  {data[0]}')
                #print (f'neuron [0]  {neuron}')
                if data[0]==str(neuron) and str(neuron)=='0':
                    avgdot = int(data[1]) / 4132
                    return avgdot
                elif data[0]==str(neuron) and str(neuron)=='1':                        
                    avgdot = int(data[1]) / 4684
                    return avgdot
                elif data[0]==str(neuron) and str(neuron)=='2':                        
                    avgdot = int(data[1]) / 4177
                    return avgdot
                elif data[0]==str(neuron) and str(neuron)=='3':                        
                    avgdot = int(data[1]) / 4351
                    return avgdot
                elif data[0]==str(neuron) and str(neuron)=='4':                        
                    avgdot = int(data[1]) / 4072
                    return avgdot
                elif data[0]==str(neuron) and str(neuron)=='5':   
                    avgdot = int(data[1]) / 3795
                    return avgdot
                elif data[0]==str(neuron) and str(neuron)=='6':                        
                    avgdot = int(data[1]) / 4137
                    return avgdot
                elif data[0]==str(neuron) and str(neuron)=='7':                        
                    avgdot = int(data[1]) / 4401
                    return avgdot
                elif data[0]==str(neuron) and str(neuron)=='8':                        
                    avgdot = int(data[1]) / 4063
                    return avgdot
                elif data[0]==str(neuron) and str(neuron)=='9':                        
                    avgdot = int(data[1]) / 4188
                    return avgdot
    
    def imagetargeted (self, imagenumber):

        rnumber = 0
        storeimage = np.array([])

        if imagenumber == 0 :
            rnumber = random.randint(2, 4133)
        elif imagenumber == 1 :
            rnumber = random.randint(4134, 8817)
        elif imagenumber == 2 :
            rnumber = random.randint(8818, 12994)
        elif imagenumber == 3:
            rnumber = random.randint(12995, 17345)      
        elif imagenumber == 4:
            rnumber = random.randint(17346, 21417)    
        elif imagenumber == 5:
             rnumber = random.randint(21418, 25212) 
        elif imagenumber == 6:
            rnumber = random.randint(25213, 29349) 
        elif imagenumber == 7:
            rnumber = random.randint(29350, 33750)
        elif imagenumber == 8:
            rnumber = random.randint(33751, 37813)    
        elif imagenumber == 9:
            rnumber = random.randint(37814, 42001)
        

        with open('./newreorg.csv', 'r') as csv_file: # probably changed to train.csv
            csvreader = csv.reader(csv_file) # loads data into csv reader
            next(csvreader) # skips pixelnumbers skips 0 
            for index, row in enumerate(csvreader): # enumerates file into indexes
               if index==rnumber:  # Load image at row given
                    pixels = row[2:] # for newreorg skips 0 as enumerate, skips 1 as label
                    #storeimage = np.append (storeimage, pixels)
                    break
        
        storeimage = np.append (storeimage, list(map(int, pixels)) ) # map characthers as integers

        return storeimage # use this to reflect image.

    def RELU(self,array): # perhaps I pass the prev output in here
                
        self.output= np.maximum(0,array)

        #!print (f'Relu {self.output}')
        return self.output
    
    def maxRELUvalue(self):
        self.maxreluval = np.max(self.output)  
        return self.maxreluval  

    def positionofmaxRELUvalue(self,apprelu_,targetsum_):

        argmax_= np.argmax(apprelu_)
        return argmax_ 
    
    def maxreluafterdot(self,weights,biases,posomaxrelu):

        neededweights  = weights[posomaxrelu].reshape(1,784) # reverse engineered     
        
        return neededweights[0]

    def newimage(self,modweights,randomimg):
        image = randomimg * modweights
        
        return image # to run image run python3 showimg.py

    def mappingwithvalues (self,imagegen,posoftarget_):
        
        desiredimage = np.array ([])

        for i, k in zip(imagegen, posoftarget_):
            #print(f'{i} -> {k}')
            if k == 0 :
                i = 0 #i * 0.05
            elif k!=0:
                i = i
            desiredimage = np.append (desiredimage,i)

        desiredimage = desiredimage.reshape(1,784)

        return desiredimage[0]

    def imageabsoluteloss (self,targetsum,imggen ):
        absoluterror =  targetsum - sum(imggen)
        print (f'sum of generated image ::: {sum(imggen)}')
        return absoluterror

    def writeimage(self,desiredimage):
    
        data = [desiredimage]

        with open('printimage.csv', 'w', encoding='UTF8', newline='') as f: 
            writer = csv.writer(f)
            writer.writerows(data)
        return data
        
    def PDRELU(self, array):

        copyofrelu = copy.deepcopy(array) # this is essential as it was getting pdrelu instead
        copyofrelu[copyofrelu<=0] = 0
        copyofrelu[copyofrelu>0] = 1
        self.derivrelu = copyofrelu

        #!print (f'self.derivrelu{self.derivrelu}')

        return self.derivrelu
    
    def inductivedotproductl1tol2 (self, inductweightsl1, inductbiasesl1):
        
        npinsertarr = np.array(self.output).reshape(10,1) # reduced this was next line : npinsertarr = npinsertarr.reshape(784,1)       
        inductiveAFTERRelu = (np.dot (inductweightsl1, npinsertarr) ) + inductbiasesl1.reshape (10,1) 
        #inductiveAFTERRelu = inductiveAFTERRelu#.reshape(10)
        #print (f'inductiveAFTERRelu::{inductiveAFTERRelu}' )

        return inductiveAFTERRelu

    def originalweightsl1tol2(self):
        return self.saveweightsl1tol2

    def originalbiasesl1tol2(self):
        return self.savebiasesl2tol1
       
    def Softmax(self,array): #! still needs fixing  

    # ! The problem is the weights.It was learning to fast thus a smaller learning rate

        self.softmaxlist =  np.exp(array) / sum(np.exp(array))
        

    #! ################### Consider deleting!!!!###############################
        #counterofzeros = 0
        #for x in range (10):
        #    if (self.softmaxlist[x]==0.0):
        #        counterofzeros+=1
        #self.nextimage = False
        #if counterofzeros == 9 : # 9 of them are zeros. This data plot is an overfit example. Bad image.
        #    self.nextimage = True
    #!######################################################################
        self.softmaxlist = self.softmaxlist.reshape(1,10)[0] 
        #!print (f'Softmax: {self.softmaxlist}')


        return self.softmaxlist

    def explodedgradientresolved(self):

        #return self.nextimage
        return 0

    def Softmaxpartialderivatives(self,array):

        soft = array.reshape(10,1)#,1)
        partials = np.diagflat(soft) - np.dot(soft, soft.T)
        return partials
        ######## Alternate Code (same output)###############
            #pdlists = []
            #newlist=[]

            #for i in range(10):
            #    same = 0
            #    rest = 0
            #    for j in range (10):
            #        if i==j:
            #            same= array[i] * (1-array[i])  # i = 0         
            #            pdlists.append(same) 
            #        else:    
            #            rest= -array[i] * (array[j])                          
            #            pdlists.append(rest) 

            #i=0
            #newlist=[]
            #while i<len(pdlists):
            #    newlist.append(pdlists[i:i+10])
            #    i+=10

            ##print (newlist)
            #return newlist
            #####################################################

    def Hotencode (self, desired):
        return np.eye(10,dtype="float")[desired] 

    def SoftmaxHotencode (self,array):

        self.predlabel = np.argmax(array)
        newarray = self.Hotencode(self.predlabel)

        return newarray

    def derivativeMAE (self,relu, yreal ):
        
        lossarray = np.array([])

        for i in range (len(relu)):
            if relu[i]==yreal:
                lossarray= np.append(lossarray, 0) # not differntialable. 
            elif relu[i] > yreal:
                lossarray= np.append (lossarray, -1)
            elif relu[i] < yreal:
                lossarray= np.append (lossarray, 1)
        
        return lossarray

    def Cross_entropy(self,correctlabel_,array) :

        nphotencode = self.Hotencode( int (correctlabel_)) #self.Hotencode(intlabel).reshape(10)  
        self.crossentropyarray = np.sum(-nphotencode*np.log(array))

        #print (f'Crossentr:::  {self.crossentropyarray}') 
        return self.crossentropyarray 

    def maxcrossentropy (self):
        self.maxnum = np.max(self.crossentropyarray)    
        return self.maxnum  
    
    def Probability(self,correctlabel_):  # pred = something new , y = label
    
        Y_ = int (correctlabel_)
        YPred_ = int (self.predlabel)
        correctpred_= 0 
        
        for i in range (10):
            if i==Y_:
                correctpred_ = self.softmaxlist[i]
                
        return (Y_==YPred_),Y_, YPred_, correctpred_ # cross entropy, if they are equal, 

    def CEntropyderivative(self,correctlabel_, softmax):


        self.crossder = np.array ([])        
        targethotencode = self.Hotencode(int(correctlabel_)) #.reshape(10).tolist() 
        
        infinitesmal = float ('-inf')
        undefined = float ('nan')    

        # ! THis is sus too will go back to this to double check. 

        for i in range (10):
            #print (f"softmax[i]{softmax[i]}") # check softmax values.if they are all 0 and one of them is 1.00 then its wrong delete that iteration
            if softmax[i] == 0.00000000e+000 or -float(targethotencode[i])/float (softmax[i]) == infinitesmal or -float(targethotencode[i])/float (softmax[i])==undefined or  -float(targethotencode[i])/float (softmax[i]) == -0. : # softmax[i] == 0.00000000e+000 or  ##reduces runtime error issue                
                self.crossder = np.append (self.crossder, 0.00)
            else:
                self.crossder = np.append(self.crossder, -float(targethotencode[i])/float (softmax[i]))

        #print (f'self.crossder{self.crossder}' )
        #print (f'new soft.crossder{-targethotencode/softmax}')

        #return -targethotencode/softmax
        return self.crossder


    def CEntropywithsoftchainrule (self,crossder,softpartials):

        loss = crossder @ softpartials 

        return loss 

    def newweightl2tol1 (self,loss,softderiv, reluapplied, kth): # also reluapplied but that is self.output


        inductPROD  = reluapplied.reshape(1,10) @ softderiv 
        inductPROD = inductPROD.reshape(10,1)
        inductPROD = inductPROD @ loss.reshape(1,10) * 1/(kth + 1) # # 10 *   10,10  * 10  =  10,10 


        # SOMETHING IS GOING ON. 
        return inductPROD  


    def newbiasl2tol1 (self,loss,softderiv, kth):
        
        baccumulator  = loss.reshape(1,10) @ softderiv 
        baccumulator = baccumulator[0] * 1/(kth+1)

        return baccumulator

    
    def newweightl1tol0(self,loss,drelu,pixels, kth ):#(self,nxtchainrule, PIXELS):

        lossdrelu = drelu.reshape(1,10) @ loss.reshape(10,1) # 1 
        variable = drelu.reshape(10,1) @ pixels.reshape (1,784) # 10 x 784
        newweightsl1tol0 = lossdrelu * variable * 1/(kth+1)

        return newweightsl1tol0

    def newbiasl1tol0 (self, loss, drelu, kth):

        #!print (f"newer drelu{drelu}")
        #print (f"newer drelu{drelu}")
        dloss = loss.reshape(1,10) @ drelu.reshape(10,1)
        biasesl1tol0 = dloss * drelu.reshape(10,1)
        biasesl1tol0 = biasesl1tol0.reshape(1,10)[0] * 1/(kth+1)
        
        return biasesl1tol0