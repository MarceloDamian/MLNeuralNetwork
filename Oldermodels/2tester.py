import generator as gg
import neuralnet as nn
import numpy as np
import pandas as pd
import csv
import random


gen = gg.GENERATE()  # calls class epoch cycle

#imgr = 3
drawnum = 4

np.random.seed (0)
w = np.random.uniform(size=(10,784),low = 0, high= 1) 
np.random.seed(1)
b = np.random.uniform(size=(1,10),low = -0.5, high= 0.5)  # very small biases

############################################################################################
np.random.seed (1)
w1 = np.random.uniform(size=(10,10),low = 0, high= 1) 
np.random.seed(2)
b1 = np.random.uniform(size=(1,10),low = -0.5, high= 0.5)  # very small biases

#############################################################################################

dw1,dw,db1,db = 0,0,0,0
boolerror = False

for i in range(0,25):#2000):#(0,3000):#1980):
    
    if i == 0:
        posofloss,boolerror,it,pw,pb,pw1,pb1,pdw,pdb,pdw1,pdb1 = gen.pixelsum(i,drawnum,w,b,w1,b1,dw,db,dw1,db1)
    else:
        posofloss,boolerror,it,pw,pb,pw1,pb1,pdw,pdb,pdw1,pdb1 = gen.pixelsum(i,drawnum,w,b,w1,b1,dw,db,dw1,db1)
        i,w,b,w1,b1,dw,db,dw1,db1 = it,pw,pb,pw1,pb1,pdw,pdb,pdw1,pdb1
        
        #print (f'weights:::  {w}, bias:::  {b}, weights1::: {w1} biases1{b1} ')
        
        if boolerror == True:
            
            break


#sumgenimage = negative     then the image is white 
#sumgenimage = positive     then the image is black



########################################################################################################################




