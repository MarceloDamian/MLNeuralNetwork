import numpy as np
import pandas as pd
import csv
import random
import copy
from sklearn.utils import shuffle
import string

class Sequential():

    def initial_deltas(self):

        pw = np.zeros((533,784)) # change these to its own function called init.
        pb = np.zeros((533,1))
        pw1 = np.zeros((533,10))
        pb1 = np.zeros((10,1))
        aw,ab,aw1,ab1,t = 0,0,0,0,0

        return pw,pb,pw1,pb1,aw,ab,aw1,ab1,t
    
    def weightsandbiases(self, layernodes):

        dynamicwandb = ()
        for i in range (0,len(layernodes)-1):
            np.random.seed (i)  # size(sets, nodes)
            w = np.random.uniform(size=(layernodes[i+1],layernodes[i]),low = -1, high= 1) * np.sqrt(2 / layernodes[i]) #maybe change to -1 as low 
            np.random.seed(i)
            b = np.random.uniform(size=(layernodes[i+1],1),low = 0, high= 1) * 0   # very small biases #b = np.random.uniform(size=(533,1),low = -1, high= 1) * np.sqrt(2 / 533) 
            dynamicwandb += (w,b,)            
        return dynamicwandb

    def shuffledtrain(self):
                
        with open('shuffledtrain.csv', 'w', encoding='UTF8', newline='') as f: 
            data = pd.read_csv('./train.csv')
            shuffled = shuffle(data,random_state=1)
            shuffleddata = np.array(shuffled)
            writer = csv.writer(f)
            writer.writerows(shuffleddata)

    def ImageArray(self): # replaces with one picture from real array 
        
        numpyarray = np.array([])

        with open('./shuffledtrain.csv', 'r') as csv_file: # probably changed to train.csv
            csvreader = csv.reader(csv_file) # loads data into csv reader
            #next(csvreader) # not needed for shuffled only for train # skips pixelnumbers skips 0 
            for index, row in enumerate(csvreader): # enumerates file into indexes
               if index==k:  # Load image at row given
                    pixels = row[1:] # for train.csv ignores or skips over the 0th column  
                    self.label = row[0] #! These are the labels of train.csv 
            
            numpyarray = np.append (numpyarray, list(map(float, pixels)) ) # map characthers as integers
            self.scaledarray = numpyarray/255
        return self.scaledarray # full image is established in nested list. Divided by 255 to get value under 1.

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

    def Linear(self,layer2nodes,layer3nodes,arraymapped,w_,b_):

        array = copy.deepcopy(arraymapped) # this is essential as it was getting pdrelu instead
        npinsertarr = np.array(array).reshape(layer2nodes,1) # reduced this was next line : npinsertarr = npinsertarr.reshape(784,1)       
        self.dotprod = (np.dot (w_, npinsertarr) ) + b_
        self.dotprod= self.dotprod.reshape(layer3nodes)
        
        return self.dotprod

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

    def LeakyRelU(self,VALUE, First_Layer): # perhaps I pass the prev output in here
        array = copy.deepcopy(First_Layer) # this is essential as it was getting pdrelu instead
        self.output= np.maximum(VALUE,array)
        return self.output
    
    def maxRELUvalue(self):
        array = copy.deepcopy(self.output) # this is essential as it was getting pdrelu instead
        self.maxreluval = np.max(array)  
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
        
    def D_LeakyRelU(self,VALUE):
        copyofrelu = copy.deepcopy(self.output) # this is essential as it was getting pdrelu instead
        copyofrelu[copyofrelu<=VALUE] = 0
        copyofrelu[copyofrelu>0] = 1
        self.derivrelu = copyofrelu

        #!print (f'self.derivrelu{self.derivrelu}')
        return self.derivrelu
    
    def originalweightsl1tol2(self):
        return self.saveweightsl1tol2

    def originalbiasesl1tol2(self):
        return self.savebiasesl2tol1
       
    def Softmax(self,Second_Layer): #! still needs fixing  
        array = copy.deepcopy(Second_Layer) # this is essential as it was getting pdrelu instead
        self.softmaxlist =  np.exp(array) / sum(np.exp(array))
        self.softmaxlist = self.softmaxlist.reshape(1,10)[0] 
        return self.softmaxlist

    def D_Softmax(self):
        array = copy.deepcopy(self.softmaxlist) # this is essential as it was getting pdrelu instead
        soft = array.reshape(10,1)#,1)
        self.partials = np.diagflat(soft) - np.dot(soft, soft.T)
        return self.partials

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

    def SoftmaxHotencode (self):
        array = copy.deepcopy(self.softmaxlist) # this is essential as it was getting pdrelu instead
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

    def CCELoss(self) :
        print (self.label)
        correctlabel_ = copy.deepcopy(self.label) # this is essential as it was getting pdrelu instead        
        array = copy.deepcopy(self.softmaxlist) # this is essential as it was getting pdrelu instead
        nphotencode = self.Hotencode( int (correctlabel_)) #self.Hotencode(intlabel).reshape(10)  
        self.crossentropyarray = np.sum(-nphotencode*np.log(array))
        print (f"CROSS ENTROPY: {self.crossentropyarray}")
        return self.crossentropyarray 

    def maxcrossentropy (self):
        self.maxnum = np.max(self.crossentropyarray)    
        print (f'CROSS ENTROPY::::: {self.maxnum}')
        return self.maxnum  
    
    def Score(self,amtcorrect):  # pred = something new , y = label
    
        correctlabel_ = copy.deepcopy(self.label) # this is essential as it was getting pdrelu instead        
        sftarray = copy.deepcopy(self.softmaxlist) # this is essential as it was getting pdrelu instead
        predlabel = np.argmax(sftarray)
        correctpred_= 0 
        
        for i in range (10):
            if i==int (correctlabel_):
                correctpred_ = sftarray[i]
        
        print (f'\nCORRECT LABEL::: {int (correctlabel_)}  PREDICTED LABEL:::  {int (predlabel)}  Probability Of Correct Label :::  {correctpred_ * 100} %\n')

        if ((int (correctlabel_)==int (predlabel)) == True ):
            amtcorrect+=1
            print (f'          TRUE: K: {k} Score: {amtcorrect/Images * 100}%   \n') 
            
        return amtcorrect#,Y_, YPred_, correctpred_ # cross entropy, if they are equal, 

    def D_CCELoss(self):
        correctlabel_ = copy.deepcopy(self.label) # this is essential as it was getting pdrelu instead        
        softmax = copy.deepcopy(self.softmaxlist) # this is essential as it was getting pdrelu instead
        self.crossder = np.array ([])        
        targethotencode = self.Hotencode(int(correctlabel_)) #.reshape(10).tolist() 
        
        infinitesmal = float ('-inf')
        undefined = float ('nan')    

        for i in range (10):
            #print (f"softmax[i]{softmax[i]}") # check softmax values.if they are all 0 and one of them is 1.00 then its wrong delete that iteration
            if softmax[i] == 0.00000000e+000 or -float(targethotencode[i])/float (softmax[i]) == infinitesmal or -float(targethotencode[i])/float (softmax[i])==undefined or  -float(targethotencode[i])/float (softmax[i]) == -0. : # softmax[i] == 0.00000000e+000 or  ##reduces runtime error issue                
                self.crossder = np.append (self.crossder, 0.00)
            else:
                self.crossder = np.append(self.crossder, -float(targethotencode[i])/float (softmax[i]))

        #print (f'self.crossder{self.crossder}' )
        return self.crossder


    def D_CCE_and_Softmax (self):
        crossder_ = copy.deepcopy(self.crossder) # this is essential as it was getting pdrelu instead
        softpartials_ = copy.deepcopy(self.partials) # this is essential as it was getting pdrelu instead
        self.loss = crossder_ @ softpartials_ 
        return self.loss 

    def Backward_Prop(self,layernodes,ogweightw1,ogweight,ogbiasb1):

        pixels = copy.deepcopy(self.scaledarray) # this is essential as it was getting pdrelu instead
        reludone = copy.deepcopy(self.output) # this is essential as it was getting pdrelu instead
        loss = copy.deepcopy(self.loss) # this is essential as it was getting pdrelu instead
        drelu = copy.deepcopy(self.derivrelu) # this is essential as it was getting pdrelu instead
        
        w1loss  = np.dot (loss.reshape(layernodes[2],1), reludone.reshape(1,layernodes[1])) * 1/(k+1)  # # 10 *   10,10  * 10  =  10,10 
        nw1 = w1loss.reshape(layernodes[2],layernodes[1])
        nb1 = np.sum(nw1,1).reshape(layernodes[2],1) * 1/(k+1)  #loss.reshape(10,1)* 1/(kth+1) 

        wloss = np.dot (loss.reshape(1,layernodes[2]),ogweightw1.reshape(layernodes[2],layernodes[1])) * drelu   
        nw = np.dot (wloss.reshape(layernodes[1],1), pixels.reshape(1,layernodes[0])) * 1/(k+1)
        nb = np.sum(nw.reshape(layernodes[1],layernodes[0]),1).reshape(layernodes[1],1) * 1/(k+1)

    
        return nw,nb,nw1,nb1 # change these to self after adding momentum 

    def GradientDescentWithMomentum(self,w,b,w1,b1,nw,nb,nw1,nb1,prevnw,prevnb,prevnw1,prevnb1,alterdw,alterdb,alterdw1,alterdb1,mu,lr):
         
        if k==0:        
            neww = w - (lr * nw) 
            newb = b - (lr * nb)     #b - (nb * learnrate)
            neww1 = w1 - (lr * nw1)  #w1 - (nw1 * learnrate) 
            newb1 = b1 - (lr * nb1)  #b1 - (nb1 * learnrate)
            alterdw,alterdb,alterdw1,alterdb1 = neww, newb, neww1, newb1
        else:

            alterdw = lr * prevnw + mu * alterdw# this is the prevalterdw
            alterdb = lr * prevnb + mu * alterdb# this is the prevalterdw
            alterdw1 = lr * prevnw1 + mu * alterdw1# this is the prevalterdw
            alterdb1 = lr * prevnb1 + mu * alterdb1# this is the prevalterdw

            neww = w - alterdw
            newb = b - alterdb
            neww1 = w1 - alterdw1
            newb1 = b1 - alterdb1

        return neww,newb,neww1,newb1,alterdw,alterdb,alterdw1,alterdb1 

    
if __name__ == "__main__":

    ################ Model #####################
    nn = Sequential()  
    pw,pb,pw1,pb1,aw,ab,aw1,ab1,t = nn.initial_deltas()
    
    Nodes = (784,533,10) # Enter node layers here
    w,b,w1,b1 = nn.weightsandbiases(Nodes)

    Images = 10 #29400 # training at 81 percent for 29,400 images. 
    Momentum = 0.9
    Learning_Rate = 0.1

    for k in range(0,Images): #trainingset:  # loops through images. 90 sec = 10 images image 0 and forward 
        imgs_ = nn.ImageArray() # K is passed through all of these functions.     
        ########################### Starting Forward Prop ###########################
        L1 = nn.Linear( Nodes[0],Nodes[1], imgs_,w,b)
        LRelu = nn.LeakyRelU(L1,0.01)  
        nn.D_LeakyRelU(0.01)
        
        L2 = nn.Linear( Nodes[1],Nodes[2], LRelu,w1,b1)
        nn.Softmax(L2) 
        ######################### Starting Backward Prop #########################
        nn.D_Softmax() # SOFTMAX partial derivatives or gradients
        nn.CCELoss() # array,labeltarget # Cross entropy on SOFTMAX  
        t = nn.Score(t)
        nn.D_CCELoss()  # array,labeltarget # Cross entropy derivative 
        nn.D_CCE_and_Softmax() # Cross entropy & SOFTMAX Chain Rule derivative

        if k!=0:
            pw,pb,pw1,pb1 = nw,nb,nw1,nb1 
        
        nw,nb,nw1,nb1 = nn.Backward_Prop(Nodes,w1,w,b1)
        w,b,w1,b1,aw,ab,aw1,ab1 = nn.GradientDescentWithMomentum(w,b,w1,b1,nw,nb,nw1,nb1,pw,pb,pw1,pb1,aw,ab,aw1,ab1,Momentum,Learning_Rate)

    accuracy = t / Images # len (trainingset) # change to len (testingset) when running testing set.    
    print (f'\n ACCURACY  ::: {accuracy * 100}%')

    # NEW ONE : 0.01 YEILDS 24 PERCENT FOR 100.
    # NEW ONE : 0.09 YEILDS 33 PERCENT FOR 100.
    # NEW ONE : 0.10 YEILDS 36 PERCENT FOR 100. For 29,400 it yeilds 80.227%
    # NEW ONE : 0.11 YEILDS 35 PERCENT FOR 100.

