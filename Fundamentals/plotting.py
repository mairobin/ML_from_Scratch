import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


##### Functional Plot

x = np.linspace(0,5,10) # array x values
y = x**2 # array y values

plt.title("Bebi Formel")
plt.xlabel("Kussis")
plt.ylabel("Bebihaftigkeit")
plt.plot(x,y)
#plt.show()


### Multiple Plots

plt.subplot(1,2,1)
plt.plot(x,y)
plt.subplot(1,2,2)
plt.plot(x,y)
plt.show()

