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
            self.scaledarray =  numpyarray/255
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

    def inductivedotproductl0tol1 (self, inductweights, inductbiases):
        arrayofpixels = copy.deepcopy(self.scaledarray) # this is essential as it was getting pdrelu instead
        npinsertarr = arrayofpixels.reshape(784,1) # reduced this was next line : npinsertarr = npinsertarr.reshape(784,1)
        #! change mutliplication to dot because it isnt supposed to just get multiplied 
        weightwithbias = np.dot (inductweights, npinsertarr) + inductbiases # maybe inductweights @ npinsertarr instead of dot. 
        self.inductivebeforerelu = weightwithbias.reshape(533) #.tolist()

        #print ("Actual:  ",self.inductivebeforerelu)
        return self.inductivebeforerelu

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

    def RELU(self): # perhaps I pass the prev output in here
        array = copy.deepcopy(self.inductivebeforerelu) # this is essential as it was getting pdrelu instead
        self.output= np.maximum(0.01,array)
        #!print (f'Relu {self.output}')
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
        
    def PDRELU(self):

        copyofrelu = copy.deepcopy(self.output) # this is essential as it was getting pdrelu instead
        copyofrelu[copyofrelu<=0.01] = 0
        copyofrelu[copyofrelu>0] = 1
        self.derivrelu = copyofrelu

        #!print (f'self.derivrelu{self.derivrelu}')

        return self.derivrelu
    
    def inductivedotproductl1tol2 (self, inductweightsl1, inductbiasesl1):
        
        array = copy.deepcopy(self.output) # this is essential as it was getting pdrelu instead
        #print (self.output)
        npinsertarr = np.array(array).reshape(533,1) # reduced this was next line : npinsertarr = npinsertarr.reshape(784,1)       
        inductweightsl1 = inductweightsl1.reshape (10,533) #! added
        self.inductiveAFTERRelu = (np.dot (inductweightsl1, npinsertarr) ) + inductbiasesl1
        #inductiveAFTERRelu = inductiveAFTERRelu#.reshape(10)
        #print (f'inductiveAFTERRelu::{inductiveAFTERRelu}' )

        return self.inductiveAFTERRelu

    def originalweightsl1tol2(self):
        return self.saveweightsl1tol2

    def originalbiasesl1tol2(self):
        return self.savebiasesl2tol1
       
    def Softmax(self): #! still needs fixing  
    # ! The problem is the weights.It was learning to fast thus a smaller learning rate
        array = copy.deepcopy(self.inductiveAFTERRelu) # this is essential as it was getting pdrelu instead
        self.softmaxlist =  np.exp(array) / sum(np.exp(array))
        self.softmaxlist = self.softmaxlist.reshape(1,10)[0] 
        #!print (f'Softmax: {self.softmaxlist}')
        return self.softmaxlist

    def Softmaxpartialderivatives(self):
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

    def Cross_entropy(self) :
        
        correctlabel_ = copy.deepcopy(self.label) # this is essential as it was getting pdrelu instead        
        array = copy.deepcopy(self.softmaxlist) # this is essential as it was getting pdrelu instead
        nphotencode = self.Hotencode( int (correctlabel_)) #self.Hotencode(intlabel).reshape(10)  
        self.crossentropyarray = np.sum(-nphotencode*np.log(array))

        #print (f'Crossentr:::  {self.crossentropyarray}') 
        return self.crossentropyarray 

    def maxcrossentropy (self):
        self.maxnum = np.max(self.crossentropyarray)    
        print (f'CROSS ENTROPY::::: {self.maxnum}')
        return self.maxnum  
    
    def Probability(self):  # pred = something new , y = label
    
        correctlabel_ = copy.deepcopy(self.label) # this is essential as it was getting pdrelu instead        
        sftarray = copy.deepcopy(self.softmaxlist) # this is essential as it was getting pdrelu instead
        Y_ = int (correctlabel_)
        YPred_ = int (self.predlabel)
        correctpred_= 0 
        
        for i in range (10):
            if i==Y_:
                correctpred_ = sftarray[i]
        
        print (f'\nCORRECT LABEL::: {Y_}  PREDICTED LABEL:::  {YPred_}  Probability Of Correct Label :::  {correctpred_ * 100} %\n')

        return (Y_==YPred_)#,Y_, YPred_, correctpred_ # cross entropy, if they are equal, 

    def CEntropyderivative(self):
        correctlabel_ = copy.deepcopy(self.label) # this is essential as it was getting pdrelu instead        
        softmax = copy.deepcopy(self.softmaxlist) # this is essential as it was getting pdrelu instead
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


    def CEntropywithsoftchainrule (self):
        crossder_ = copy.deepcopy(self.crossder) # this is essential as it was getting pdrelu instead
        softpartials_ = copy.deepcopy(self.partials) # this is essential as it was getting pdrelu instead
        self.loss = crossder_ @ softpartials_ 
        return self.loss 

    def errorbackprop(self,ogweightw1,ogweight,ogbiasb1,kth):
        derivsft = copy.deepcopy(self.partials) # this is essential as it was getting pdrelu instead
        pixels = copy.deepcopy(self.scaledarray) # this is essential as it was getting pdrelu instead
        reludone = copy.deepcopy(self.output) # this is essential as it was getting pdrelu instead
        loss = copy.deepcopy(self.loss) # this is essential as it was getting pdrelu instead
        drelu = copy.deepcopy(self.derivrelu) # this is essential as it was getting pdrelu instead
        cross = copy.deepcopy(self.crossder) # this is essential as it was getting pdrelu instead

        
        w1loss  = np.dot (loss.reshape(10,1), reludone.reshape(1,533)) * 1/(kth + 1) # # 10 *   10,10  * 10  =  10,10 
        nw1 = w1loss.reshape(533,10)
        #################### everythings good. I am a bit concerned about bias...####################
        nb1 = np.sum(nw1.reshape(10,533),1).reshape(10,1) * 1/(kth+1)  #loss.reshape(10,1)* 1/(kth+1) 

        wloss = np.dot (loss.reshape(1,10),ogweightw1.reshape(10,533)) * drelu   
        nw = np.dot (wloss.reshape(533,1), pixels.reshape(1,784)) * 1/(kth + 1)
        
        nb = np.sum(nw.reshape(533,784),1).reshape(533,1) * 1/(kth+1) 
        
        return nw,nb,nw1,nb1


    def updatewandb(self,w,b,w1,b1,nw,nb,nw1,nb1,kth,mu,lr):
     
        #print (nb, "nbshape", nb.shape)

        # w = w – change_x  # what you have 
        # change_x = lr * nw
        #############################################

        #change_x(t) = lr * nw(t-1) + mu * (lr * nw)o(t-1)   # what you want. Desired
        
        #w(t) = w(t-1) – change_x(t)
        ###Done: ### w(0) =  w - (nw * lr) 

        # We need the previous weight or bias. 
        # We need the previous lr value. 
        # we need to set a condition for w(0)

        #if kth==0 :        
        neww = w - (lr * nw) 
        newb = b - (lr * nb)    #b - (nb * learnrate)
        neww1 = w1 - (lr * nw1) #w1 - (nw1 * learnrate) 
        newb1 = b1 - (lr * nb1) #b1 - (nb1 * learnrate)
        #else:
        #    neww = prevw - (lr * prevnw + mu * prevlr * prevnw) 
        #    newb = prevb - (lr * prevnb + mu * prevlr * prevnb)     #b - (nb * learnrate)
        #    neww1 = prevw1 - (lr * prevnw1 + mu * prevlr * prevnw1)     #w1 - (nw1 * learnrate) 
        #    newb1 = prevb1 - (lr * prevnb1 + mu * prevlr * prevnb1) #b1 - (nb1 * learnrate)


        #change_x(t) = lr * nw(t-1) + mu * (lr * nw)o(t-1)   # what you want. Desired
        
        #w(t) = w(t-1) – change_x(t)

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

    
if __name__ == "__main__":
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


    actispred = False
    accuratepredictions = 0

    up_to = 100

    for k in range(0,up_to): #trainingset:  # loops through images. 90 sec = 10 images image 0 and forward 

        ######### initalizing data  #################
        Fullcycle= EpochCycle()  # calls class epoch cycle
        Fullcycle.replacewithrealarray(k) #(3)# read all 42000. For loop. Gets data from index 4         
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
        nw,nb,nw1,nb1 = Fullcycle.errorbackprop(w1,w,b1,k)
        w,b,w1,b1= Fullcycle.updatewandb(w,b,w1,b1,nw,nb,nw1,nb1,k,0.9, 0.23)

        # 0.01 goes down 
        # 0.1 yeilds 26 percent for 100. 31.9 percent for 1000 images
        # 0.2 yields 31 percent for 100. 39.9% for 1000 images. 61 percent for 29,400.
        # 0.22 yeilds 32 percent for 100. 41.4 percent for 1000 images
        # 0.23 yeilds 33 percent for 100.  42.9 percent for 1000 images    
        # 0.24 yeilds 32 percent for 100.  yeilds 43.3 percent for 1000
        # 0.25 yeilds 42.9 percent for 1000.
        #EDIT: FOR 0.2 IT YEILDS 40 percent for 1000 images

        print (k)
        actispred,k,w,b,w1,b1 = actispred,k,w,b,w1,b1

        if actispred == True:
            print (f' ACCURATE == PREDICTED   TRUE   K: {k}   \n') 
            accuratepredictions+=1
            #break 

    accuracy = accuratepredictions / up_to # len (trainingset) # change to len (testingset) when running testing set.    
    print (f'\n ACCURACY  ::: {accuracy * 100}%')

    #print ("hello")