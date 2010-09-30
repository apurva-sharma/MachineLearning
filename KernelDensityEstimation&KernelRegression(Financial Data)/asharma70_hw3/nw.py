import numpy as np
import scipy

import numpy,math
from numpy import *
from collections import defaultdict
from math import log, sqrt
from matplotlib.ticker import LinearLocator, FixedLocator, FormatStrFormatter
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pylab as P

import random
def filereader(filename):
    """
    Reads a filename and returns a string of contents
    """
    f=open(filename)
    contents=f.read()
    f.close()
    return contents


def gaussian_kernel(u):
    power = (u**2)/2
    return numpy.exp(power)/(sqrt(2* numpy.pi))



def calc_euclidean_distance(training_coordinates,query_coordinates):
    distance=sqrt(reduce(lambda x,y:x+y, map(lambda x,y:pow(x-y,2),training_coordinates,query_coordinates)))
    return distance

def multidim_kernel_estimate(x,xi,h):
    '''
    xi is one point of the set of all points
    x is the test point
    '''
    
    x = array(x)
    xi = array(xi)
    difference = (x-xi)/h

    results=[]

    for i in difference:
        results.append((gaussian_kernel(i))/h)

    result = reduce(lambda x,y:x*y,results)

    return result


def parseTimeSeries(contents,windowsize):
    start = 0
    temp = []
    while (True):
        end = start+windowsize
        if end >= len(contents):
            break
        temp.append([contents[start:end],contents[end]])
        start+=1
    return temp


def normalize(li):
    ma = max(li)
    mi = min(li)
    a = map(lambda x: (x-mi)/ma,li)
    return a


def calculate_weights(x,points,h):
    '''
    returns a row of n elements signifying the weights for
    each of the n points
    '''
    operand_list = points
##    for i in points:
##        operand_list.append(multidim_kernel_estimate(x,i,h))
##        #operand_list.append(calc_euclidean_distance(i,x))

    weight_list = []
    for i in operand_list:
        #num = gaussian_kernel(i)
        num = multidim_kernel_estimate(x,i,h)
        den = 0
        for j in range(0,len(points)):
            #den_operand = calc_euclidean_distance(x,points[j])
            #den_operand = calc_euclidean_distance(x,points[j])
            den+= multidim_kernel_estimate(x,points[j],h)
            #den += gaussian_kernel(den_operand)

        weight = num/float(den)
        weight_list.append(weight)
    return weight_list
        
                            
h = 20
c = filereader('sp500_small.csv')
parsed_contents1 = map(lambda x:float(x),c.splitlines())

parsed_contents = parsed_contents1
#parsed_contents = normalize(parsed_contents1)
parsed_dataset = parseTimeSeries(parsed_contents,10)
features = map(lambda x:x[0],parsed_dataset)
values = map(lambda x:x[1],parsed_dataset)
#test = features[0]



distances = []
##gaussian_returns = []
##for i in features:
##    d = calc_euclidean_distance(i,test)
##    distances.append(d)
##    gaussian_returns.append(gaussian_kernel(d))
##
##num =gaussian_kernel(calc_euclidean_distance(test,features[7]))
##den = reduce(lambda x,y:x+y,gaussian_returns)
##w = num/den
##r = w*  values[7]


chosen_indices = []
test_points =[]
##for i in range(0,5):
##    r = random.randint(0,len(features))
##    chosen_indices.append(r)
##    test_points.append(features[r])
    


test_points = features

predictions = []

for test in test_points:
    w1 = calculate_weights(test,features,h)
    prediction = 0
    for i in range(0,len(w1)):
        prediction += (w1[i]*values[i])
    predictions.append(prediction)

st = 'prediction,actual\n'
for i in range(0,len(predictions)):
    st+=str(predictions[i])+','+str(values[i])+'\n'

predictions = array(predictions)
values = array(values)
x = arange(0,len(values))

fig = P.figure()

P.plot(x, predictions, 'o')
P.plot(x, values, '^')

P.show()

f = open('predictionsh50.csv','w')
f.write(st)
f.close()

