import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt



def randrange(n, vmin, vmax):
    return (vmax-vmin)*np.random.rand(n) + vmin

fig = plt.figure()
ax = Axes3D(fig)
#ax = fig.add_subplot(111, projection='3d')
n = 100
#for c, m, zl, zh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
for c, m, zl, zh in [('r', 'o', -50, -25),('b', '^', -30, -5)]:
    xs = [1,2,9,4,5]
    ys = [4,5,3,3,5]
    zs = [7,5,3,7,5]
    ax.scatter(xs, ys, zs, c=c, marker=m)




ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

