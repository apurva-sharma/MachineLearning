import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

mpl.rcParams['legend.fontsize'] = 10

fig = plt.figure()
ax =Axes3D(fig)
#fig.gca(projection='3d')
theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
z = [5,5,5,5]
x = [1,2,3,4]
y = [1,2,1,1]
ax.plot(x,'r--', y, z)
ax.legend()

plt.show()
