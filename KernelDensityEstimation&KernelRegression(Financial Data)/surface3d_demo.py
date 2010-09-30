from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FixedLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

##fig = plt.figure()
ax = Axes3D(fig)
X = np.arange(-2, 2, 1)	
Y = np.arange(-2, 2, 1)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)

#Z = X+Y

Z = np.sin(R)

print Z
print Z.shape
print X.shape
print Y.shape
##surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet,
##        linewidth=0, antialiased=False)
#ax.set_zlim3d(-1.01, 1.01)

#ax.w_zaxis.set_major_locator(LinearLocator(10))
#ax.w_zaxis.set_major_formatter(FormatStrFormatter('%.03f'))

#fig.colorbar(surf, shrink=0.5, aspect=5)

##plt.show()

