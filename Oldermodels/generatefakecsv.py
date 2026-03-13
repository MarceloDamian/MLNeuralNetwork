
#finished
import numpy as np
from random import randrange
import csv
import random

header = ['pixelnumbers']
lista = []

# perhaps have all the black numbers converge to the edges by training it as well.

for num in range(784):  # basis step 
    lista = lista + [randrange(256),]
data = [lista]

for z in range(27999): #(27999): # add 1 to get the total as the previous for loop added a line # inductive step
    x = data 
    y = list(x[0]) # changes dataype to change tuples that are immutable
    for i in range(len(y)):
        y[i] = randrange(256) # changes each individual element # this is where you would add the weight or bias 
        x = list(y) 
    n = [x]
    data = data + n


with open('faketest.csv', 'w', encoding='UTF8', newline='') as f:
#with open('newimage.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(header)

    # write multiple rows
    writer.writerows(data)

