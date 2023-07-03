import matplotlib.pyplot as plt
import numpy as np
import torch
np.random.seed(111)


# Feature reduction
x1 = np.linspace(0,10,10)
x2 = np.linspace(10,20,10)
y1 = x1 * 0.1 + np.random.normal(0, 0.1, 10)
y2 = x2 * 0.1 + np.random.normal(0, 0.1, 10)
plt.scatter(x1,y1)
plt.scatter(x2,y2)
plt.ylim([0,2.5])
plt.plot(np.linspace(00,20,20),np.linspace(00,20,20)*0.1, color='r')
plt.plot(np.linspace(00,20,20),np.linspace(00,20,20)*-0.2 + 3, color='r')
plt.xticks([])
plt.yticks([])

plt.show()