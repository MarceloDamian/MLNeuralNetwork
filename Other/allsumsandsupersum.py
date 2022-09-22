
import neuralnet as nn
import numpy as np
import pandas as pd
import csv
import random


classfirst= nn.Zerolayer()  #layer0 = nn.InputLayer(42000, 784) #layer0 = nn.InputLayer(784, 10) # danwrote
arrayzero = classfirst.firstnodes(1,784)


summedproduct = 0
header = ['summations']
supersummed = []



for i in range (0,42000):
    realarr = classfirst.replacewithrealarray(i)# change later on will be added a for loop to map thru all 42000
#    print (realarr)
    summed = classfirst.summedoutputofanyarray(0,realarr) # bias added and realarray
    #print (f"dotproduct of the real first one with added bias {summed}\n")
    supersummed.append (summed)
    #print (supersummed)

    summedproduct+=summed
    print (f"index : {i} summedoutput: {summedproduct}")
    

data=[supersummed]



with open('supersum.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(header)

    # write multiple rows
    writer.writerows(data)
