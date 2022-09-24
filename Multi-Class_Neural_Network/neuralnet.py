import numpy as np
import pandas as pd
import csv
import random
import copy
from sklearn.utils import shuffle
import string

class Sequential():

    def init_delta(self, ):
        variables = (len(Nodes)-1) * 2  # the amount of pw's and aw's plus t, 1 , to count the score. 
        repeat = []  # Empty tuple.
        for i in range (variables):
            pw = 0
            repeat += [pw,]
        return repeat

    def weightsandbiases(self):

        dynamicwandb = []
        for i in range (0,len(Nodes)-1):
            np.random.seed (i)  # size(sets, nodes)
            w = np.random.uniform(size=(Nodes[i+1],Nodes[i]),low = -1, high= 1) * np.sqrt(2 / Nodes[i]) #maybe change to -1 as low 
            np.random.seed(i)
            b = np.random.uniform(size=(Nodes[i+1],1),low = 0, high= 1) * 0   # very small biases #b = np.random.uniform(size=(533,1),low = -1, high= 1) * np.sqrt(2 / 533) 
            dynamicwandb += [w,b,]  
        #print (dynamicwandb)
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

    
    def Linear(self,layer2nodes,layer3nodes,arraymapped,w_,b_):

        array = copy.deepcopy(arraymapped) # this is essential as it was getting pdrelu instead
        npinsertarr = np.array(array).reshape(layer2nodes,1) # reduced this was next line : npinsertarr = npinsertarr.reshape(784,1)       
        dotprod = (np.dot (w_, npinsertarr) ) + b_
        dotprod= dotprod.reshape(layer3nodes)
        
        return dotprod

    def LeakyRelU(self,VALUE, First_Layer): # perhaps I pass the prev output in here
        array = copy.deepcopy(First_Layer) # this is essential as it was getting pdrelu instead
        self.output= np.maximum(VALUE,array)
        return self.output
    
    def D_LeakyRelU(self,VALUE):
        copyofrelu = copy.deepcopy(self.output) # this is essential as it was getting pdrelu instead
        copyofrelu[copyofrelu<=VALUE] = 0
        copyofrelu[copyofrelu>0] = 1
        self.derivrelu = copyofrelu

        #!print (f'self.derivrelu{self.derivrelu}')
        return self.derivrelu
    
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

    def Hotencode (self, desired):
        return np.eye(10,dtype="float")[desired] 

    def CCELoss(self) :
        print (self.label)
        correctlabel_ = copy.deepcopy(self.label) # this is essential as it was getting pdrelu instead        
        array = copy.deepcopy(self.softmaxlist) # this is essential as it was getting pdrelu instead
        nphotencode = self.Hotencode( int (correctlabel_)) #self.Hotencode(intlabel).reshape(10)  
        self.crossentropyarray = np.sum(-nphotencode*np.log(array))
        print (f"CROSS ENTROPY: {self.crossentropyarray}")
        return self.crossentropyarray 

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

    def Leaky_Relu_BackProp(self,drelu2_,relu1_,chainloss_,ogweightw2_): # Generalize so you can then add dynamically

        relu1 = copy.deepcopy(relu1_) # this is essential as it was getting pdrelu instead
        chainloss = copy.deepcopy(drelu2_) # this is essential as it was getting pdrelu instead
        drelu2 = copy.deepcopy(ogweightw2_) # this is essential as it was getting pdrelu instead
        # Relu2 to Relu1
        wloss2 = np.dot (chainloss.reshape(1,Nodes[2]), ogweightw2.reshape(Nodes[2],Nodes[1])) * drelu2           
        nw2 = np.dot (wloss2.reshape(Nodes[1],1), relu1.reshape(1,Nodes[0])) * 1/(k+1)
        nb2 = np.sum(nw.reshape(Nodes[1],Nodes[0]),1).reshape(Nodes[1],1) * 1/(k+1)                
        return nw2,nb2

    def Backward_Prop(self):

        pixels = copy.deepcopy(self.scaledarray) # this is essential as it was getting pdrelu instead
        reludone = copy.deepcopy(self.output) # this is essential as it was getting pdrelu instead
        Chain = copy.deepcopy(self.loss) # this is essential as it was getting pdrelu instead
        drelu = copy.deepcopy(self.derivrelu) # this is essential as it was getting pdrelu instead
        
        # Softmax to Relu 
        w1loss  = np.dot (Chain.reshape(Nodes[2],1), reludone.reshape(1,Nodes[1])) 
        self.nw1 = w1loss.reshape(Nodes[2],Nodes[1]) * 1/(k+1)  
        self.nb1 = np.sum(self.nw1,1).reshape(Nodes[2],1) * 1/(k+1)  

        # Relu to Relu 

        # Relu To Pixels
        wloss = np.dot (Chain.reshape(1,Nodes[2]), WandB[2].reshape(Nodes[2],Nodes[1])) * drelu           
        self.nw = np.dot (wloss.reshape(Nodes[1],1), pixels.reshape(1,Nodes[0])) * 1/(k+1)
        self.nb = np.sum(self.nw.reshape(Nodes[1],Nodes[0]),1).reshape(Nodes[1],1) * 1/(k+1)

    
        return self.nw,self.nb,self.nw1,self.nb1 # change these to self after adding momentum 

    def GradientDescentWithMomentum(self,mu,lr):#aw,ab,aw1,ab1,mu,lr):#,w,b,w1,b1,nw,nb,nw1,nb1,prevnw,prevnb,prevnw1,prevnb1,alterdw,alterdb,alterdw1,alterdb1,mu,lr):
        
        for i in range (len(Nodes) + 1):  # 0,1,2,3 
            if k==0:
                OptWandB[i] = WandB[i] - (lr * NWandB[i])#self.nw) 
                Altvalues[i] = OptWandB[i]
            else:
                Altvalues[i] = lr * Prevvalues[i] + mu * Altvalues[i]#? Works the same. 
                OptWandB[i] = WandB[i] - Altvalues[i] # Becomes new weights   
            
        return OptWandB, Altvalues

if __name__ == "__main__":
    
    ################# Model #####################
    
    Nodes = (784,533,10) #(784,533,356,10)  # Enter node layers here

    nn = Sequential() 
    Prevvalues, Altvalues, NWandB, OptWandB = nn.init_delta(), nn.init_delta(), nn.init_delta(), nn.init_delta()
    WandB = nn.weightsandbiases() # Both of these dynamically grow.

    t = 0
    Images = 10     #29400 # training at 81 percent for 29,400 images. 
    Momentum = 0.9
    Learning_Rate = 0.1

    for k in range(0,Images): #trainingset:  # loops through images. 90 sec = 10 images image 0 and forward 
        
        imgs_ = nn.ImageArray() # K is passed through all of these functions.     
        ########################### Starting Forward Prop ###########################
        
        L1 = nn.Linear( Nodes[0], Nodes[1], imgs_,WandB[0],WandB[1])
        L1Relu = nn.LeakyRelU(L1,0.01)  
        nn.D_LeakyRelU(0.01)

        ######################### Starting Backward Prop #########################
        L3 = nn.Linear( Nodes[1], Nodes[2], L1Relu, WandB[2],WandB[3])  #! change nodes to 2 and 3 and l1relu to l2relu

        nn.Softmax(L3) 
        nn.D_Softmax() # SOFTMAX partial derivatives or gradients
        nn.CCELoss() # array,labeltarget # Cross entropy on SOFTMAX  
        t = nn.Score(t)
        nn.D_CCELoss()  # array,labeltarget # Cross entropy derivative 
        softloss = nn.D_CCE_and_Softmax() # Cross entropy & SOFTMAX Chain Rule derivative

        if k!=0:
            Prevvalues = NWandB # store previous weights and biases. 
            
        NWandB = nn.Backward_Prop()
        WandB, Altvalues = nn.GradientDescentWithMomentum(Momentum,Learning_Rate)    

    accuracy = t / Images # len (trainingset) # change to len (testingset) when running testing set.    
    print (f'\n ACCURACY  ::: {accuracy * 100}%')

    # NEW ONE : 0.01 YEILDS 24 PERCENT FOR 100.
    # NEW ONE : 0.09 YEILDS 33 PERCENT FOR 100.
    # NEW ONE : 0.10 YEILDS 36 PERCENT FOR 100. For 29,400 it yeilds 80.227%
    # NEW ONE : 0.11 YEILDS 35 PERCENT FOR 100.
    # imgs, relu , relu , softmax
    #     L1,    L2,  , L3

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