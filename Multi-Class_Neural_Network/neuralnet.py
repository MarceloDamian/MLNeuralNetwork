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
        
        npinsertarr = np.array(arraymapped).reshape(layer2nodes,1) # reduced this was next line : npinsertarr = npinsertarr.reshape(784,1)       
        dotprod = (np.dot (w_, npinsertarr) ) + b_
        dotprod = dotprod.reshape(layer3nodes)
        
        return dotprod

    def LeakyRelU(self,First_Layer,VALUE): # perhaps I pass the prev output in here 
        return np.maximum(VALUE,First_Layer)

    def D_LeakyRelU(self,PREVLEAKY,VALUE):
        return np.greater(PREVLEAKY, VALUE).astype(int)
    
    def Softmax(self,Second_Layer): #! still needs fixing  

        self.softmaxlist =  np.exp(Second_Layer) / sum(np.exp(Second_Layer))
        self.softmaxlist = self.softmaxlist.reshape(1,10)[0] 
        return self.softmaxlist

    def D_Softmax(self):
        soft = self.softmaxlist.reshape(10,1)
        self.partials = np.diagflat(soft) - np.dot(soft, soft.T)
        return self.partials

    def Hotencode (self, desired):
        return np.eye(10,dtype="float")[desired] 

    def CCELoss(self) :       
        nphotencode = self.Hotencode( int (self.label)) #self.Hotencode(intlabel).reshape(10)  
        self.crossentropyarray = np.sum(-nphotencode*np.log(self.softmaxlist))
        print (f"CROSS ENTROPY: {self.crossentropyarray}")
        return self.crossentropyarray 

    def Score(self,amtcorrect):  # pred = something new , y = label
    
        predlabel = np.argmax(self.softmaxlist)
        percentpred_= 0 
        
        for i in range (10):
            if i == int(self.label):
                percentpred_ = self.softmaxlist[i]

        print (f'\nCORRECT LABEL::: {int (self.label)}  PREDICTED LABEL:::  {int (predlabel)}  Probability Of Correct Label :::  {percentpred_ * 100} %\n')

        if ((int (self.label)==int (predlabel)) == True ):
            amtcorrect+=1
            print (f'          TRUE: K: {k} Score: {amtcorrect/Images * 100}%   \n') 
            
        return amtcorrect

    def D_CCELoss(self):

        self.crossder = np.array ([])        
        targethotencode = self.Hotencode(int(self.label)) #.reshape(10).tolist() 
        infinitesmal = float ('-inf')
        undefined = float ('nan')   

        for i in range (10):
            DCCE = -float(targethotencode[i])/float (self.softmaxlist[i])
            if DCCE == (infinitesmal or undefined or -0.) : # 0.00000000e+000or  ##reduces runtime error issue                
                self.crossder = np.append (self.crossder, 0.00)
            else:
                self.crossder = np.append(self.crossder, -float(targethotencode[i])/float (self.softmaxlist[i]))
        
        return self.crossder

    def D_CCE_and_Softmax (self):
        self.loss = self.crossder @ self.partials 
        return self.loss 

    def Leaky_Relu_BackProp(self,drelu2_,relu2_): # Generalize so you can then add dynamically

        relu2 = copy.deepcopy(relu2_) # this is essential as it was getting pdrelu instead
        drelu = copy.deepcopy(drelu2_) # this is essential as it was getting pdrelu instead
        
        # Relu2 to Relu1 0 ,2,4
        #self.loss = 10 , 
        wloss = np.dot (self.loss, WandB[2].reshape(Nodes[2],Nodes[1])) * drelu           
        self.neww = np.dot (wloss.reshape(Nodes[1],1), relu2.reshape(1,Nodes[0])) * 1/(k+1)
        self.newb = np.sum(self.neww.reshape(Nodes[1], Nodes[0]),1).reshape(Nodes[1],1) * 1/(k+1) 
                      
        return self.neww,self.newb

    def Backward_Prop(self,PREVLEAKY,DL1Relu):

        #reludone = copy.deepcopy(PREVLEAKY) # this is essential as it was getting pdrelu instead
        #drelu = copy.deepcopy(DL1Relu) # this is essential as it was getting pdrelu instead
        
        # Softmax to Relu 
        w1loss  = np.dot (self.loss.reshape(Nodes[2],1), PREVLEAKY.reshape(1,Nodes[1])) 
        nw1 = w1loss.reshape(Nodes[2],Nodes[1]) * 1/(k+1)  
        nb1 = np.sum(nw1,1).reshape(Nodes[2],1) * 1/(k+1)  
        # Relu To Pixels
        wloss = np.dot (self.loss.reshape(1,Nodes[2]), WandB[2].reshape(Nodes[2],Nodes[1])) * DL1Relu           
        nw = np.dot (wloss.reshape(Nodes[1],1), self.scaledarray.reshape(1,Nodes[0])) * 1/(k+1)
        nb = np.sum(nw.reshape(Nodes[1],Nodes[0]),1).reshape(Nodes[1],1) * 1/(k+1)
        #wloss2 = np.dot (self.loss.reshape(1,Nodes[2]), WandB[4].reshape(Nodes[2],Nodes[2])) * DL1Relu           
        #self.neww = np.dot (wloss2.reshape(Nodes[2],1), self.scaledarray.reshape(1,Nodes[1])) * 1/(k+1)
        #self.newb = np.sum(self.neww.reshape(Nodes[2],Nodes[1]),1).reshape(Nodes[2],1) * 1/(k+1) 
        # self.loss dot w2  * drelu  dot  pixels
        return nw,nb,nw1,nb1#self.neww,self.newb,nw1,nb1 

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
    
    Nodes = (784,533,10) 
    #Nodes = (784,533,356,10)  # Enter node layers here

    nn = Sequential() 
    Prevvalues, Altvalues, NWandB, OptWandB = nn.init_delta(), nn.init_delta(), nn.init_delta(), nn.init_delta()
    WandB = nn.weightsandbiases() # Both of these dynamically grow.

    t = 0
    Images = 100     #29400 # training at 81 percent for 29,400 images. 
    Momentum = 0.9
    Learning_Rate = 0.1


    for k in range(0,Images): #trainingset:  # loops through images. 90 sec = 10 images image 0 and forward 
        
        imgs_ = nn.ImageArray() # K is passed through all of these functions.     
        ########################### Starting Forward Prop ###########################
        
        L1 = nn.Linear( Nodes[0], Nodes[1], imgs_,WandB[0],WandB[1])
        L1Relu = nn.LeakyRelU(L1,0.01)  
        DL1Relu = nn.D_LeakyRelU(L1Relu,0.01)
        ######################### Starting Backward Prop #########################

        #L2 = nn.Linear( Nodes[1], Nodes[2], L1Relu,WandB[2],WandB[3])
        #L2Relu = nn.LeakyRelU(L2,0.01)  
        #DL2Relu = nn.D_LeakyRelU(L2Relu,0.01)
        #L3 = nn.Linear( Nodes[2], Nodes[3], L2Relu, WandB[4],WandB[5])  #! change nodes to 2 and 3 and l1relu to l2relu

        L3 = nn.Linear( Nodes[1], Nodes[2], L1Relu, WandB[2],WandB[3])  #! change nodes to 2 and 3 and l1relu to l2relu

        nn.Softmax(L3) 
        nn.D_Softmax() # SOFTMAX partial derivatives or gradients
        nn.CCELoss() # array,labeltarget # Cross entropy on SOFTMAX  
        t = nn.Score(t)
        nn.D_CCELoss()  # array,labeltarget # Cross entropy derivative 
        softloss = nn.D_CCE_and_Softmax() # Cross entropy & SOFTMAX Chain Rule derivative

        if k!=0:
            Prevvalues = NWandB # store previous weights and biases. CHANGE TO PLUS AND =.
        
        #test = nn.Leaky_Relu_BackProp(DL2Relu,L2Relu)
       
        NWandB = nn.Backward_Prop(L1Relu,DL1Relu)
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