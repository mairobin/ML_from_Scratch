import numpy as np

# Vectors





# Matrix Multiplication

m1 = np.matrix([[0,-2,2],[5,1,5],[1,4,-1]])
m2 = np.matrix([[0,1,2],[3,4,5],[6,7,8]])

print(m1)
#print(m1[2][2]) # DOES NOT WORK

print(m1[0,1]) # Count from 0

res1 = m1*m2
res2 = m1@m2
res3 = m1.dot(m2)

print()

