import numpy as np
import pandas as pd
import csv
import random
import copy
from sklearn.utils import shuffle



class EpochCycle():
    
    def shuffledtrain(self):
                
        with open('shuffledtrain.csv', 'w', encoding='UTF8', newline='') as f: 
            data = pd.read_csv('./train.csv')
            shuffled = shuffle(data,random_state=1)
            shuffleddata = np.array(shuffled)
            writer = csv.writer(f)
            writer.writerows(shuffleddata)

    def replacewithrealarray(self, rnumber): # replaces with one picture from real array 
        
        numpyarray = np.array([])

        with open('./shuffledtrain.csv', 'r') as csv_file: # probably changed to train.csv
            csvreader = csv.reader(csv_file) # loads data into csv reader
            #next(csvreader) # not needed for shuffled only for train # skips pixelnumbers skips 0 
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

    def inductivedotproductl0tol1 (self, inductweights, inductbiases, arrayofpixels):
        
        npinsertarr = arrayofpixels.reshape(784,1) # reduced this was next line : npinsertarr = npinsertarr.reshape(784,1)

        #! change mutliplication to dot because it isnt supposed to just get multiplied 
        weightwithbias = np.dot (inductweights, npinsertarr) + inductbiases # maybe inductweights @ npinsertarr instead of dot. 
        inductivebeforeRelu = weightwithbias.reshape(533) #.tolist()

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
        self.output= np.maximum(0.01,array)
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
        copyofrelu[copyofrelu<=0.01] = 0
        copyofrelu[copyofrelu>0] = 1
        self.derivrelu = copyofrelu

        #!print (f'self.derivrelu{self.derivrelu}')

        return self.derivrelu
    
    def inductivedotproductl1tol2 (self, inductweightsl1, inductbiasesl1):
        
        npinsertarr = np.array(self.output).reshape(533,1) # reduced this was next line : npinsertarr = npinsertarr.reshape(784,1)       
        inductweightsl1 = inductweightsl1.reshape (10,533) #! added
        inductiveAFTERRelu = (np.dot (inductweightsl1, npinsertarr) ) + inductbiasesl1
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
        self.softmaxlist = self.softmaxlist.reshape(1,10)[0] 
        #!print (f'Softmax: {self.softmaxlist}')
        return self.softmaxlist

    def Softmaxpartialderivatives(self,array):

        soft = array.reshape(10,1)#,1)
        partials = np.diagflat(soft) - np.dot(soft, soft.T)
        return partials

    #def Softmaxpartialderivatives(self,array):
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

        crossder = np.array ([])        
        targethotencode = self.Hotencode(int(correctlabel_)) #.reshape(10).tolist() 
        
        infinitesmal = float ('-inf')
        undefined = float ('nan')    

        # ! THis is sus too will go back to this to double check. 

        for i in range (10):
            #print (f"softmax[i]{softmax[i]}") # check softmax values.if they are all 0 and one of them is 1.00 then its wrong delete that iteration
            if softmax[i] == 0.00000000e+000 or -float(targethotencode[i])/float (softmax[i]) == infinitesmal or -float(targethotencode[i])/float (softmax[i])==undefined or  -float(targethotencode[i])/float (softmax[i]) == -0. : # softmax[i] == 0.00000000e+000 or  ##reduces runtime error issue                
                crossder = np.append (crossder, 0.00)
            else:
                crossder = np.append(crossder, -float(targethotencode[i])/float (softmax[i]))

        #print (f'self.crossder{self.crossder}' )
        #print (f'new soft.crossder{-targethotencode/softmax}')

        #return -targethotencode/softmax
        return crossder


    def CEntropywithsoftchainrule (self,crossder,softpartials):

        loss = crossder @ softpartials 

        #print (f"cechain = {loss}")

        return loss 

    def errorbackprop(self,derivsft, pixels,reludone,loss,drelu,cross,ogweightw1,ogweight,ogbiasb1,kth):
         
        w1loss  = np.dot (loss.reshape(10,1), reludone.reshape(1,533)) * 1/(kth + 1) # # 10 *   10,10  * 10  =  10,10 
        nw1 = w1loss.reshape(533,10)
        #################### everythings good. I am a bit concerned about bias...####################
        nb1 = np.sum(nw1.reshape(10,533),1).reshape(10,1) * 1/(kth+1)  #loss.reshape(10,1)* 1/(kth+1) 

        wloss = np.dot (loss.reshape(1,10),ogweightw1.reshape(10,533)) * drelu   
        nw = np.dot (wloss.reshape(533,1), pixels.reshape(1,784)) * 1/(kth + 1)
        
        nb = np.sum(nw.reshape(533,784),1).reshape(533,1) * 1/(kth+1) 
        
        return nw,nb,nw1,nb1


    def updatewandb(self,w,b,w1,b1,nw,nb,nw1,nb1,kth,momentum,lr):
     
        #print (nb, "nbshape", nb.shape)

        # w = w – change_x  # what you have 
        # change_x = lr * nw
        #############################################

        # change_x(t) = lr * nw(t-1) + mu * (lr * nw)o(t-1)   # what you want. Desired
                
        # w(t) = w(t-1) – change_x(t)
        # w(0) =  w - (nw * lr) 


        ####### Yeilds 20 percent. STABLE.
        neww = w - (nw * lr) 
        newb = b - (nb * lr)#b - (nb * learnrate)
        neww1 = w1 - (nw1 * lr) #w1 - (nw1 * learnrate) 
        newb1 = b1 - (nb1 * lr)#b1 - (nb1 * learnrate)

        #print(nb1 * learnrate)
        #print ("\nthis one ",b, "i=1  ")
        # b is lessrandom. and 
        #print (w, "first  W")
        #print (b, "first  B")
        #print (w1, "first  W1")
        #print (b1, "first B1")
        #print (dw, "first DW")
        #print (db, "first DB")
        #print (dw1, "first DW1")
        #print (db1, "first DB1")

            
        return neww,newb,neww1,newb1

    
