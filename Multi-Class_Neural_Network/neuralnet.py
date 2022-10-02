import csv, random, numpy as np, pandas as pd
import numpy as np, pandas as pd
from sklearn.utils import shuffle
from scipy.special import log_softmax

class Sequential():
    def init_var(self): # Initializes array to zeros.
        return [0] * (Iter_Index * 2)
    def init_hiddenlayers(self): # Initializes array to object 'None'.
        return np.empty((len(Nodes)- 2), dtype=object)
    def weightsandbiases(self): # Initalizes weights and biases using He's initialization.
        dynamicwandb = []
        for i in range (0,Iter_Index):
            np.random.seed (i) 
            w = np.random.uniform(size=(Nodes[i+1],Nodes[i]),low = -1, high= 1) * np.sqrt(2 / Nodes[i]) #maybe change to -1 as low 
            dynamicwandb += [w,0] 
        return dynamicwandb

    def shuffledtrain(self): # Shuffles original training data. 
        with open('shuffledtrain.csv', 'w', encoding='UTF8', newline='') as f: 
            data = pd.read_csv('./train.csv')
            shuffled = shuffle(data,random_state=1)
            shuffleddata = np.array(shuffled)
            writer = csv.writer(f)
            writer.writerows(shuffleddata)

    def ImageArray(self): # Maps each individual 28x28 image in training and divides it by its max rgb value 255. 
        numpyarray = np.array([])
        with open('./shuffledtrain.csv', 'r') as csv_file: # probably changed to train.csv
            csvreader = csv.reader(csv_file) # loads data into csv reader
            for index, row in enumerate(csvreader): # enumerates file into indexes
               if index==k:  # Load image at row given
                    pixels = row[1:] # for train.csv ignores or skips over the 0th column  
                    self.label = row[0] #! These are the labels of train.csv 
            self.scaledarray = np.append (numpyarray, list(map(float, pixels))) / 255 # map characthers as integers
        return self.scaledarray # full image is established in nested list. Divided by 255 to get value under 1.

    def Linear(self,layer2nodes,layer3nodes,arraymapped,w_,b_): # Uses dot product on images using weights and biases.
        npinsertarr = np.array(arraymapped).reshape(layer2nodes,1)
        return (np.dot (w_, npinsertarr)  + b_ ).reshape(layer3nodes) 

    def LeakyRelU(self,First_Layer,VALUE):  # filters dot prod with leaky relu value.
        return np.maximum(VALUE,First_Layer)

    def D_LeakyRelU(self,PREVLEAKY,VALUE): # Derivative of leaky relu values using true or false values.
        return np.greater(PREVLEAKY, VALUE).astype(int)
    
    def Softmax(self,Second_Layer):  # Numericaly stable softmax for high softmax values. 
        exp = np.exp(Second_Layer - max(Second_Layer))
        self.softmaxlist = (exp /np.sum(exp))
        return self.softmaxlist

    def D_Softmax(self): # derivative of Softmax values using diagflat/jacobian matrix and dot prod. 
        soft = self.softmaxlist.reshape(Nodes[Iter_Index],1)
        self.partials = np.diagflat(soft) - np.dot(soft, soft.T)
        return self.partials

    def Hotencode (self, desired): # uses np.eye to hotencode array. 
        return np.eye(10,dtype="float")[desired] 

    def C_CrossEntropyLoss(self): # categorical cross entropy of softmax using numerically stable log_softmax
        targethotencode = self.Hotencode( int (self.label)) 
        self.crossentropyarray = np.sum(-targethotencode*log_softmax(self.softmaxlist))
        print (f"CROSS ENTROPY: {self.crossentropyarray}")
        return self.crossentropyarray

    def Score(self, t_): # Finds accuracy score. 
        predlabel = np.argmax(self.softmaxlist) # index of max in self.softmaxlist array     
        percentpred_ = self.softmaxlist[int(self.label)] # percent prdeiction using accuratelabel.
        acuratebool = (int(self.label)==int (predlabel)) # bool if prediction = accurate label 
        print (f'\nCORRECT LABEL::: {int (self.label)}  PREDICTED LABEL:::  {int (predlabel)}  Probability Of Correct Label :::  {percentpred_ * 100} %\n')
        t_= t+1 if acuratebool==True else t_  # increments score if prediction = true value.
        print (f'\t\t\tTRUE: K: {k} Score: {t_/Images * 100}%\t\n') if acuratebool==True else 0 # prints score if prediction = true value.
        return t_

    def D_CCELoss(self): # derivative of categorical cross entropy
        self.crossder = np.array ([])        
        targethotencode = self.Hotencode(int(self.label)) 
        for i in range (10):
            if self.softmaxlist[i] == 0.00: # fixed runtime issue as division by 0 was not possible to significant digit 0.01.
                self.crossder = np.append(self.crossder, -float(targethotencode[i])/float (0.01))
            elif self.softmaxlist[i] != 0.00: # Allows derivative of categorical cross entropy with given softmax value.
                self.crossder = np.append(self.crossder, -float(targethotencode[i])/float (self.softmaxlist[i]))
        return self.crossder

    def D_CCE_and_Softmax (self): # chain rule applied to softmax and categorical cross entropy
        self.loss = self.D_Softmax() @ self.D_CCELoss() 
        return self.loss 

    def helper_loss_prop (self,floor):
        ArrayIndex = Iter_Index*2-2 # Index of WandB. -2 to account for the array index. 
        wloss = np.dot (self.D_CCE_and_Softmax().reshape(1,Nodes[Iter_Index]), WandB[ArrayIndex].reshape(Nodes[Iter_Index],Nodes[Iter_Index-1])) * DERIV[len(Nodes)-3]  #First loss for weights and bias   
        for decrement in range (Iter_Index-1,floor,-1):
            wloss = np.dot (wloss.reshape(1,Nodes[decrement]),WandB[ArrayIndex-2].reshape(Nodes[decrement],Nodes[decrement-1])) * DERIV[decrement -2]  # Next loss running through for loop
            ArrayIndex-= 2 # indexing through weights.
        return wloss
    
    def helper_wandb_prop (self,start,clock,startminusclock,array, wloss):
        LayerWandB[start]= wloss.reshape (Nodes[startminusclock],1) * array.reshape(1, Nodes[clock+1]) * 1/(k+1) # Helps map loss over weights.
        LayerWandB[start+1] = np.sum(LayerWandB[start],1).reshape(Nodes[startminusclock],1) * 1/(k+1) # Helps map loss over biases.
        return LayerWandB[start], LayerWandB[start+1] # returns backproped weights and biases.

    def Backward_Prop(self,ACTIV,DERIV): # Backwards propagates using error values. 

        clock,floor = 0,2 # clock is a counter that increments through the for loops. Floor is also updated to increment. 
        LayerWandB[0],LayerWandB[1] = self.helper_wandb_prop(0,-1,1,self.scaledarray, self.helper_loss_prop(1)) # finds first weights and biases. 

        for next_ in range (2, Iter_Index*2 , 2): # loops through all weights and biases.
            if len(Nodes)!=3:
                wloss = self.helper_loss_prop(floor) # calls helper function to change wloss.
            if next_ == Iter_Index*2-2:
                wloss = self.loss.reshape(Nodes[len(Nodes)-1],1)  #If it arrives at the last index then set it to the categorical cross entropy loss
            LayerWandB[next_],LayerWandB[next_+1]=self.helper_wandb_prop(next_,clock,int (next_-clock),ACTIV[clock], wloss)# calls helper function and uses parameters appropriately.
            floor+=1 
            clock+=1    
        return LayerWandB[0: Iter_Index*2] # From 0 to max iterations

    def GradientDescentWithMomentum(self,mu,lr):

        for i in range (len(Nodes) + Hiddenlayers):  # makes sure to optimize all weights and biases
            if k==0: # first one doesnt have previous momentum. 
                OptWandB[i] = WandB[i] - (lr * NWandB[i])# gradient descent. 
                Altvalues[i] = OptWandB[i] # essential to loop through. 
            else:
                Altvalues[i] = lr * Prevvalues[i] + mu * Altvalues[i] # adds momentum using previous values and updates accordingly. 
                OptWandB[i] = WandB[i] - Altvalues[i]  # gradient descent. 

        return OptWandB, Altvalues # Altvalues are values previous passed over using momentum except for the first iteration. 

if __name__ == "__main__":
    
    ################# Model #####################
    Nodes = (784,533,10)  # Feel free to add as many layers. Each number represents amount of nodes in its respected layer. (layer0,layer1,...layeri)
    Iter_Index, Hiddenlayers = len(Nodes) - 1, len(Nodes) - 2 # To index arrays. Amount of hidden layers.

    nn = Sequential() # Class name. 
    Prevvalues, Altvalues, NWandB, OptWandB, LayerWandB = nn.init_var(), nn.init_var(), nn.init_var(), nn.init_var(),nn.init_var() # initailizes variables to 0.
    ACTIV,DERIV = nn.init_hiddenlayers(), nn.init_hiddenlayers() # Initializes activ = activation function, deriv = derivative of activation function to 0. 
    WandB = nn.weightsandbiases() # He's intialization for weights and biases
    
    #Images = 35700 # Training dataset 80 percent o 42,000
    #Images = 35700,42001 # 3600 images Test dataset 20 percent o 42,000
    Images = 10 
    range_ = range(Images)
    
    t = 0 # tracks score
    Momentum = 0.9 # Momentum's defult is 0.9
    Learning_Rate =0.1 # Best learning rate for 3 layer nn is 0.1. Smaller for more layers

    for k in range_: 

        imgs_ = nn.ImageArray()    # images mapped by rgb values 255.
        
        L0 = nn.Linear( Nodes[0], Nodes[1], imgs_, WandB[0],WandB[1]) # dot prod on weights and biases. WANDB[even] = weights, WANDB[ODD] = biases,
        ACTIV[0] = nn.LeakyRelU(L0,0.01)  # ACTIV = activation function. Starting at index 0.
        DERIV[0] = nn.D_LeakyRelU(ACTIV[0],0.01) # Deriv  = Derivative of activation function.  Starting at index 0.
        
        L1 = nn.Linear( Nodes[1], Nodes[2], ACTIV[0],WandB[2],WandB[3])

        nn.Softmax(L1) # Numerically stable Softmax function of previous layer
        nn.C_CrossEntropyLoss() # Numerically Stable Loss/Cost function 
        t = nn.Score(t) # SCORE

        Prevvalues = NWandB if k!=0 else 0  # Gathers previous weights and biases for momentum.
        NWandB = nn.Backward_Prop ( ACTIV, DERIV ) # Backward propagation. 
        WandB, Altvalues = nn.GradientDescentWithMomentum(Momentum,Learning_Rate)  # Optimization function with given momentum and learning rate. 

    accuracy = t / Images  # True positive + True negative / All images. Accuracy score.
    print (f'\n ACCURACY  ::: {accuracy * 100}%')

    # Learning rate : 0.10 YEILDS 36 PERCENT FOR 100. For 29,400 it yeilds 80.227% For a 3 Layer NN. (784,533,10)
    
    ######## Alternate Code (same output)###############
        #def Softmaxpartialderivatives(self,array):

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
        