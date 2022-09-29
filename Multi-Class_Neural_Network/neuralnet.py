import numpy as np
import pandas as pd
import csv
import random
import copy
from sklearn.utils import shuffle
import string

class Sequential():

    def init_var(self):
        return [0] * (MaxConnects * 2)
    def init_hiddenlayers(self):
        return np.empty((len(Nodes)- 2), dtype=object)
    def weightsandbiases(self):
        dynamicwandb = []
        for i in range (0,MaxConnects):
            np.random.seed (i)  # size(sets, nodes)
            w = np.random.uniform(size=(Nodes[i+1],Nodes[i]),low = -1, high= 1) * np.sqrt(2 / Nodes[i]) #maybe change to -1 as low 
            dynamicwandb += [w,0,] 
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
        self.softmaxlist = ( np.exp(Second_Layer) / sum(np.exp(Second_Layer)) ).reshape(1,10)[0]
        return self.softmaxlist#.reshape(1,10)[0]

    def D_Softmax(self):
        soft = self.softmaxlist.reshape(Nodes[MaxConnects],1)
        self.partials = np.diagflat(soft) - np.dot(soft, soft.T)
        return self.partials

    def Hotencode (self, desired):
        return np.eye(10,dtype="float")[desired] 

    def CCELoss(self) :       
        nphotencode = self.Hotencode( int (self.label)) #self.Hotencode(intlabel).reshape(10)  
        self.crossentropyarray = np.sum(-nphotencode*np.log(self.softmaxlist))
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
        targethotencode = self.Hotencode(int(self.label)) #.reshape(10).tolist() 
        infinitesmal = float ('-inf')
        undefined = float ('nan')   

        for i in range (Nodes[MaxConnects]):
            DCCE = -float(targethotencode[i])/float (self.softmaxlist[i])
            if DCCE == (infinitesmal or undefined or -0.) :               
                self.crossder = np.append (self.crossder, 0.00)
            else:
                self.crossder = np.append(self.crossder, -float(targethotencode[i])/float (self.softmaxlist[i]))
        
        return self.crossder
 
    def D_CCE_and_Softmax (self):
        self.loss = self.crossder @ self.partials 
        return self.loss 

    def Backward_Prop(self,ACTIV,DERIV):#leakyrelu, DL1Relu, Lastleakyrelu, Dl2Relu):
    
        hiddenlayers = len(Nodes) - 2 #print (MaxConnects)#print (hiddenlayers)

        print (hiddenlayers, "Hiddenlayers")
        print (MaxConnects, "MaxConnects")

        #Nodes = (784,533,356,122,10) 0,1,2,3,4
        #WandB= w,b,w1,b1,w2,b2,w3,b3
        #w = 533,784 # change these to its own function called init.
        #b = 533,1
        #w1 = 356,533
        #b1 = 356, 1
        #w2 = 122,356
        #b2 = 122, 1
        #w3 = 10,122
        #b3 = 10,1

        # Do 6 layers and comment it out 
        # Do 5 layers and comment it out. 
        # Do 4 layers and comment it out         
        # do 3 layers and comment it out 

        ##################### FOR 6 TOTAL LAYERS ############################
        #######################################################################

        ##################### FOR 5 TOTAL LAYERS ############################
        #######################################################################

        ##################### FOR 4 TOTAL LAYERS ############################
        #######################################################################

        ##################### FOR 3 TOTAL LAYERS ############################
        wloss = np.dot (self.loss.reshape(1,Nodes[2]), WandB[2].reshape(Nodes[2],Nodes[1])) * DERIV[0]    
        LayerWandB[0] = wloss.reshape(Nodes[1],1)  * self.scaledarray.reshape(1,Nodes[0]) * 1/(k+1) # 533,784
        LayerWandB[1] = np.sum(LayerWandB[0],1).reshape(Nodes[1],1) * 1/(k+1) # 533,1
        w1loss  = self.loss.reshape(Nodes[2],1) *  ACTIV[0].reshape(1,Nodes[1]) # 3N = 0 , 4N = 2
        LayerWandB[2] = w1loss.reshape(Nodes[2],Nodes[1]) * 1/(k+1)  # 10,356
        LayerWandB[3] = np.sum(LayerWandB[2],1).reshape(Nodes[2],1) * 1/(k+1)   # 10,1
        ###############################################################
        
        return LayerWandB[0: AllConnects]  # self.neww,self.newb,Lastw,Lastb 



    def GradientDescentWithMomentum(self,mu,lr):

        for i in range (len(Nodes) + 1):  # 0,1,2,3 
            if k==0:
                OptWandB[i] = WandB[i] - (lr * NWandB[i])
                Altvalues[i] = OptWandB[i]
            else:
                Altvalues[i] = lr * Prevvalues[i] + mu * Altvalues[i]#? Works the same. 
                OptWandB[i] = WandB[i] - Altvalues[i] # Becomes new weights   
            
        return OptWandB, Altvalues

if __name__ == "__main__":
    
    # So this works for three layers. However, I am attempting to get it to work for 4 layers and essentially making it to do it dynamically.

    ################# Model #####################
    
    Nodes = (784,533,10) 
    #Nodes = (784,533,356,10)  # Enter node layers here
    #Nodes = (784,533,356,122,10) 


    MaxConnects = len(Nodes) - 1
    AllConnects = MaxConnects * 2 


    nn = Sequential() 
    Prevvalues, Altvalues, NWandB, OptWandB, LayerWandB = nn.init_var(), nn.init_var(), nn.init_var(), nn.init_var(),nn.init_var()
    WandB = nn.weightsandbiases() # Both of these dynamically grow.
    ACTIV = nn.init_hiddenlayers() 
    DERIV = nn.init_hiddenlayers() 
    

    t = 0
    Images = 10     #29400 # training at 81 percent for 29,400 images. 
    Momentum = 0.9
    Learning_Rate = 0.1# 0.05 #0.1

    for k in range(0,Images): #trainingset:  # loops through images. 90 sec = 10 images image 0 and forward 
        
        imgs_ = nn.ImageArray() # K is passed through all of these functions.     
        
        ########################### Starting Forward Prop ###########################
        L1 = nn.Linear( Nodes[0], Nodes[1], imgs_, WandB[0],WandB[1])

        ACTIV[0] = nn.LeakyRelU(L1,0.01)  
        DERIV[0] = nn.D_LeakyRelU(ACTIV[0],0.01)
        
        ######################### Starting Backward Prop #########################
        #L2 = nn.Linear( Nodes[1], Nodes[2], ACTIV[0],WandB[2],WandB[3])
        #ACTIV[1] = nn.LeakyRelU(L2,0.01)  
        #DERIV[1] = nn.D_LeakyRelU(ACTIV[1],0.01)
        
        #L3 = nn.Linear( Nodes[2], Nodes[3], ACTIV[1], WandB[4],WandB[5])  #! change nodes to 2 and 3 and l1relu to l2relu
        
        
        
        #ACTIV[2] = nn.LeakyRelU(L3,0.01)  
        #DERIV[2] = nn.D_LeakyRelU(ACTIV[2],0.01)
        #L4 = nn.Linear( Nodes[3], Nodes[4], ACTIV[2], WandB[6],WandB[7])  #! change nodes to 2 and 3 and l1relu to l2relu




        L3 = nn.Linear( Nodes[1], Nodes[MaxConnects], ACTIV[0], WandB[2],WandB[3])  #! Actual
        #nn.Softmax(L4)
        nn.Softmax(L3) 
        nn.D_Softmax() # SOFTMAX partial derivatives or gradients
        nn.CCELoss() # array,labeltarget # Cross entropy on SOFTMAX  
        t = nn.Score(t)
        nn.D_CCELoss()  # array,labeltarget # Cross entropy derivative 
        softloss = nn.D_CCE_and_Softmax() # Cross entropy & SOFTMAX Chain Rule derivative

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








         # wloss = np.dot (self.loss.reshape(1,Nodes[MaxConnects]), WandB[hiddenlayers*2].reshape(Nodes[MaxConnects],Nodes[MaxConnects-1])) * DERIV[hiddenlayers-1]    
        # if hiddenlayers > 1:

        #     wloss  = np.dot (wloss.reshape(1,Nodes[MaxConnects-1]),  WandB[hiddenlayers].reshape(Nodes[MaxConnects-1],Nodes[MaxConnects-2]) ) * DERIV[0]#DERIV[1] # 1,533

        #     if hiddenlayers >=3:
        #         wloss  = np.dot (wloss.reshape(1,Nodes[MaxConnects-2]),  WandB[2].reshape(Nodes[MaxConnects-2],Nodes[MaxConnects-3]) ) * DERIV[0] # 1,533


        #     for j in range (2, len(Nodes)-1):# 2 for 2h # 2,3 for 3h  # 2,3,4 for 4h

        #         newwloss = np.dot (self.loss.reshape(1,Nodes[MaxConnects]),WandB[hiddenlayers*2].reshape(Nodes[MaxConnects],Nodes[MaxConnects-1])) * DERIV[len(Nodes)-1 - j] # 1,356
                
        #         if j == 2:
        #             LayerWandB[2]= newwloss.reshape (Nodes[MaxConnects-1],1) * ACTIV[0].reshape(1, Nodes[MaxConnects-2]) * 1/(k+1) # 356,533 # 4
        #             LayerWandB[3] = np.sum(LayerWandB[2],1).reshape(Nodes[MaxConnects-1],1) * 1/(k+1) # 356,1      # 3 # 2 ,1 
                
        #         if j==3:
        #             LayerWandB[j+1]= newwloss.reshape (Nodes[MaxConnects-1],1) * ACTIV[0].reshape(1, Nodes[MaxConnects-2]) * 1/(k+1) # 356,533 # 4
                
        #         if j > 3:
        #             LayerWandB[j+2]= newwloss.reshape (Nodes[MaxConnects-1],1) * ACTIV[0].reshape(1, Nodes[MaxConnects-2]) * 1/(k+1) # 356,533 # 4
        #             LayerWandB[j+1] = np.sum(LayerWandB[j+2],1).reshape(Nodes[MaxConnects-1],1) * 1/(k+1) # 356,1      # 3 # 2 ,1 
            
        
        # ##################### Same Below ############################
        # LayerWandB[0] = wloss.reshape(Nodes[1],1)  * self.scaledarray.reshape(1,Nodes[0]) * 1/(k+1) # 533,784
        # LayerWandB[1] = np.sum(LayerWandB[0],1).reshape(Nodes[1],1) * 1/(k+1) # 533,1
        # ###############################################################

        # # Softmax to Relu 
        # ##################### Same Below ############################
        # w1loss  = self.loss.reshape(Nodes[MaxConnects],1) *  ACTIV[hiddenlayers-1].reshape(1,Nodes[MaxConnects-1]) # 3N = 0 , 4N = 2
        # LayerWandB[AllConnects-2] = w1loss.reshape(Nodes[MaxConnects],Nodes[MaxConnects-1]) * 1/(k+1)  # 10,356
        # LayerWandB[AllConnects-1] = np.sum(LayerWandB[AllConnects-2],1).reshape(Nodes[MaxConnects],1) * 1/(k+1)   # 10,1
        # ###############################################################