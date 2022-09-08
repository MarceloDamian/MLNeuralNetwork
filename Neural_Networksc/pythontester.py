import numpy as np


p = np.eye(1,dtype="float") # 0 returns nothing. 
#print (p)
x = np.array ([-1,2,3,4,5,8,-1,-2,3,4,5,-20])
x[x<=0] = 0
x[x>0] = 1


a = [[1, 0], [0, 1]]
b = [[4, 1], [2, 2] ]     

# 1*4 + 0*2    1*1 + 0*2
# 0*4 + 1*2    0*1 + 1*2 
# 4 , 1
# 2 , 2

np.dot(a, b)

print (np.dot(a, b) )

a = [1,2,3]
b = [3,34,5]

print (np.dot(a,b))
