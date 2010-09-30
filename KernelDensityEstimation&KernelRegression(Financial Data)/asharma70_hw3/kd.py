from math import sqrt,fabs
import numpy,math
from numpy import *
from collections import defaultdict
from math import log, sqrt

from matplotlib.ticker import LinearLocator, FixedLocator, FormatStrFormatter
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import cm
import sys,os

def filereader(filename):
    """
    Reads a filename and returns a string of contents
    """
    f=open(filename)
    contents=f.read()
    f.close()
    return contents



def parse_file(contents):
    """
    Accepts a string of contents and returns a structure of format [[[x1,x2,...xn],y],...]
    """
    holder=[]
    split_contents=contents.splitlines()
    #split_contents.append(
    temp=[]
    temp1=[]
    for i in split_contents:
        temp1=i.split(',')[0:2]
        temp1= map(lambda x:float(x),temp1)
        temp.append(temp1)

    return temp




def gaussian_kernel(u):
    power = (u**2)/2
    return numpy.exp(power)/(sqrt(2* numpy.pi))

def normalize(data):
    data = array(data)
    mins=[]
    maxes=[]
    for i in range(0,len(data[0])):
        mins.append(min(map(lambda x:x[i],data)))
        maxes.append(max(map(lambda x:x[i],data)))

    normalized_data=[]
    #print "mins", mins
    #print "maxes",maxes
    for i in range(0,len(data)):
        for j in range(0,len(data[0])):
            data[i][j]=(data[i][j]-mins[j])/maxes[j]
    
    
    return data


def bivariate_kernel_estimate(x,xi,h):
    '''
    xi is one point of the set of all points
    x is the test point
    '''
    
    x = array(x)
    xi = array(xi)
    difference = (x-xi)/h

    t1 = difference[0]
    t2 = difference[1]
    
    
    t1_result = (gaussian_kernel(t1))/h
    t2_result = (gaussian_kernel(t2))/h
    result = (t1_result*t2_result)/h

    return result



def univariate_kernel_estimate(x,xi,h):

   # print "x ",x,"xi ",xi
    difference = (x-xi[0])/h
    result = (gaussian_kernel(difference))/h
    return result


    
h = .8
f = 'iris.data'

c = filereader(f)
p = parse_file(c)
n = normalize(p)



x1s = map(lambda x:x[0],n)
x2s = map(lambda x:x[1],n)

test_point_generator = arange(0,1,.05)

test_points=[]

for i in test_point_generator:
    test_points.append([i,i])
x1 = array(test_point_generator)
y1 = sin(test_point_generator)*2

def getDensities(X,Y,n,h):
    zs= zeros(shape(X))
    for i in range(0,len(X)):
        for j in range(0,len(X[0])):
            test_point = [X[i][j],Y[i][j]]
            density = 0
            for k in range(0,len(n)):
                density+=bivariate_kernel_estimate(test_point,n[k],h)
            density/=len(n)
            zs[i][j] = density

    return zs
            
            
            
    
fig = plt.figure()
ax = Axes3D(fig)
X = array(x1)
Y = array(y1)
X, Y = numpy.meshgrid(X, Y)
Z = getDensities(X,Y,n,h)


surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet,
        linewidth=0, antialiased=False)
ax.set_zlim3d(-1.01, 1.01)

ax.w_zaxis.set_major_locator(LinearLocator(10))
ax.w_zaxis.set_major_formatter(FormatStrFormatter('%.03f'))

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()



