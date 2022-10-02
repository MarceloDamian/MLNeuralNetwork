import numpy as np
import pandas as pd
import csv
import random
from sklearn.utils import shuffle
from scipy.special import log_softmax

class Sequential():
    def init_var(self):
        return [0] * (Iter_Index * 2)
    def init_hiddenlayers(self):
        return np.empty((len(Nodes)- 2), dtype=object)
    def weightsandbiases(self):
        dynamicwandb = []
        for i in range (0,Iter_Index):
            np.random.seed (i)  # size(sets, nodes)
            w = np.random.uniform(size=(Nodes[i+1],Nodes[i]),low = -1, high= 1) * np.sqrt(2 / Nodes[i]) #maybe change to -1 as low 
            dynamicwandb += [w,0] 
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
            for index, row in enumerate(csvreader): # enumerates file into indexes
               if index==k:  # Load image at row given
                    pixels = row[1:] # for train.csv ignores or skips over the 0th column  
                    self.label = row[0] #! These are the labels of train.csv 
            self.scaledarray = np.append (numpyarray, list(map(float, pixels))) / 255 # map characthers as integers
        return self.scaledarray # full image is established in nested list. Divided by 255 to get value under 1.

    def Linear(self,layer2nodes,layer3nodes,arraymapped,w_,b_):
        npinsertarr = np.array(arraymapped).reshape(layer2nodes,1) # reduced this was next line : npinsertarr = npinsertarr.reshape(784,1)   
        return (np.dot (w_, npinsertarr)  + b_ ).reshape(layer3nodes) 

    def LeakyRelU(self,First_Layer,VALUE): # perhaps I pass the prev output in here 
        return np.maximum(VALUE,First_Layer)

    def D_LeakyRelU(self,PREVLEAKY,VALUE):
        return np.greater(PREVLEAKY, VALUE).astype(int)
    
    def Softmax(self,Second_Layer): 
        exp = np.exp(Second_Layer - max(Second_Layer))
        self.softmaxlist =exp /np.sum(exp)
        return self.softmaxlist#.reshape(1,10)[0]

    def D_Softmax(self):
        soft = self.softmaxlist.reshape(Nodes[Iter_Index],1)
        self.partials = np.diagflat(soft) - np.dot(soft, soft.T)
        return self.partials

    def Hotencode (self, desired):
        return np.eye(10,dtype="float")[desired] 

    def C_CrossEntropyLoss(self) :       
        nphotencode = self.Hotencode( int (self.label)) #self.Hotencode(intlabel).reshape(10)  
        self.crossentropyarray = np.sum(-nphotencode*log_softmax(self.softmaxlist))#np.log(self.softmaxlist))
        print (f"CROSS ENTROPY: {self.crossentropyarray}")
        return self.crossentropyarray

    def Score(self, t_):  # pred = something new , y = label

        predlabel = np.argmax(self.softmaxlist)      
        percentpred_ = self.softmaxlist[int(self.label)]
        acuratebool = (int(self.label)==int (predlabel))
        print (f'\nCORRECT LABEL::: {int (self.label)}  PREDICTED LABEL:::  {int (predlabel)}  Probability Of Correct Label :::  {percentpred_ * 100} %\n')
        t_= t+1 if acuratebool==True else t_  
        print (f'\t\t\tTRUE: K: {k} Score: {t_/Images * 100}%\t\n') if acuratebool==True else 0
        return t_

    def D_CCELoss(self):
        self.crossder = np.array ([])        
        targethotencode = self.Hotencode(int(self.label)) 
        for i in range (10):
            if self.softmaxlist[i] == 0.00:    
                self.crossder = np.append(self.crossder, -float(targethotencode[i])/float (0.01))
            elif self.softmaxlist[i] != 0:  
                self.crossder = np.append(self.crossder, -float(targethotencode[i])/float (self.softmaxlist[i]))
        return self.crossder

    def D_CCE_and_Softmax (self):
        self.loss = self.D_Softmax() @ self.D_CCELoss()
        return self.loss 

    def Backward_Prop(self,ACTIV,DERIV):#leakyrelu, DL1Relu, Lastleakyrelu, Dl2Relu):
    
        ArrayIndex = Iter_Index*2-2
        storearray = ArrayIndex
        clock,floor = 0,2
        wloss = np.dot (self.D_CCE_and_Softmax().reshape(1,Nodes[Iter_Index]), WandB[ArrayIndex].reshape(Nodes[Iter_Index],Nodes[Iter_Index-1])) * DERIV[len(Nodes)-3]    
                    
        for decrement in range (Iter_Index-1, 1, -1 ): # so 4,3,2
            wloss = np.dot (wloss.reshape(1,Nodes[decrement]), WandB[ArrayIndex-2].reshape(Nodes[decrement],Nodes[decrement-1 ])) * DERIV[decrement-2]    
            ArrayIndex-= 2

        LayerWandB[0] = wloss.reshape(Nodes[1],1)  * self.scaledarray.reshape(1,Nodes[0]) * 1/(k+1)
        LayerWandB[1] = np.sum(LayerWandB[0],1).reshape(Nodes[1],1) * 1/(k+1)
        ArrayIndex = storearray

        for next_ in range (2, Iter_Index*2 , 2):
            
            if len(Nodes)!=3:
                wloss = np.dot (self.D_CCE_and_Softmax().reshape(1,Nodes[Iter_Index]), WandB[ArrayIndex].reshape(Nodes[Iter_Index],Nodes[Iter_Index-1])) * DERIV[len(Nodes)-3]    
                for decrement in range (Iter_Index-1,floor,-1):
                    wloss = np.dot (wloss.reshape(1,Nodes[decrement]),WandB[ArrayIndex-2].reshape(Nodes[decrement],Nodes[decrement-1])) * DERIV[decrement -2]      
                    ArrayIndex-= 2
            if next_ == Iter_Index*2-2:
                wloss  = self.loss.reshape(Nodes[len(Nodes)-1],1)  # changed to len(nodes)

            LayerWandB[next_]= wloss.reshape (Nodes[next_ - clock ],1) * ACTIV[clock].reshape(1, Nodes[clock+1]) * 1/(k+1) # should be 356,533 not 122,533
            LayerWandB[next_+1] = np.sum(LayerWandB[next_],1).reshape(Nodes[next_- clock],1) * 1/(k+1) 
            floor+=1
            clock+=1    
            ArrayIndex = storearray
        
        return LayerWandB[0: Iter_Index*2]  

    def GradientDescentWithMomentum(self,mu,lr):

        for i in range (len(Nodes) + Hiddenlayers):  # 0,1,2,3 
            if k==0:
                OptWandB[i] = WandB[i] - (lr * NWandB[i])
                Altvalues[i] = OptWandB[i]
            else:
                Altvalues[i] = lr * Prevvalues[i] + mu * Altvalues[i]#? Works the same. 
                OptWandB[i] = WandB[i] - Altvalues[i] # Becomes new weights   

        return OptWandB, Altvalues

if __name__ == "__main__":
    
    ################# Model #####################
    Nodes = (784,533,10)  
    
    Iter_Index, Hiddenlayers = len(Nodes) - 1, len(Nodes) - 2

    nn = Sequential() 
    Prevvalues, Altvalues, NWandB, OptWandB, LayerWandB = nn.init_var(), nn.init_var(), nn.init_var(), nn.init_var(),nn.init_var()
    ACTIV,DERIV = nn.init_hiddenlayers(), nn.init_hiddenlayers() 
    WandB = nn.weightsandbiases() 
    
    t = 0
    Images = 100     #29400 # training at 81 percent for 29,400 images. 
    Momentum = 0.9
    Learning_Rate =0.1

    for k in range(0,Images): #trainingset:  # loops through images. 90 sec = 10 images image 0 and forward 
        
        imgs_ = nn.ImageArray() # K is passed through all of these functions.     
        
        L1 = nn.Linear( Nodes[0], Nodes[1], imgs_, WandB[0],WandB[1])
        ACTIV[0] = nn.LeakyRelU(L1,0.01)  
        DERIV[0] = nn.D_LeakyRelU(ACTIV[0],0.01)
        
        L2 = nn.Linear( Nodes[1], Nodes[2], ACTIV[0],WandB[2],WandB[3])

        nn.Softmax(L2) 
        nn.C_CrossEntropyLoss()
        t = nn.Score(t)

        Prevvalues = NWandB if k!=0 else 0  # store previous weights and biases. CHANGE TO PLUS AND =.        
        NWandB = nn.Backward_Prop ( ACTIV, DERIV )
        WandB, Altvalues = nn.GradientDescentWithMomentum(Momentum,Learning_Rate)   

    accuracy = t / Images # len (trainingset) # change to len (testingset) when running testing set.    
    print (f'\n ACCURACY  ::: {accuracy * 100}%')

    # NEW ONE : 0.01 YEILDS 24 PERCENT FOR 100.
    # NEW ONE : 0.09 YEILDS 33 PERCENT FOR 100.
    # NEW ONE : 0.10 YEILDS 36 PERCENT FOR 100. For 29,400 it yeilds 80.227%
    # NEW ONE : 0.11 YEILDS 35 PERCENT FOR 100.
    # imgs, relu , relu , softmax
    #     L1,    L2,  , L3

    #0.02 for more hidden layers it yeilds better results 
    #Nodes = (784,533,356 ,10) 0,1,2,3
    # WandB= w,b,w1,b1,w2,b2
    #w = 533,784 # change these to its own function called init.
    #b = 533,1
    #neww = 356,533
    #newb = 356, 1
    #w1 = 10,356
    #b1 = 10,1

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
       

    
    #Nodes = (784,533,10) 3# #0123       4 # wandb 0,1  ,2,3  # NEEDS 4 
    #Nodes = (784,533,356,10) 4# #01234     5 # wandb 0,1  ,2,3  ,4,5  # NEEDS 6

    #Nodes = (784,533,356,122,10) 5# #012345     6 # wandb 0,1  ,2,3  ,4,5, 6,7  # NEEDS 8 
    #Nodes = (784,533,356,122,50,10) 6# #0123456     7 # wandb 0,1  ,2,3  ,4,5, 6,7  8,9 # NEEDS 10 
    # 3, 4  1
    # 4, 6  2
    # 5, 8  3
    # 6, 10  4 
    
    #hidlayers = len(Nodes) - 2