#!/usr/bin/env python
"""
Author :Apurva Sharma
email  :asharma70@gatech.edu
GTID   :902490301
Simple em algorithm
usage : python em.py <train file> <k- optional (default 3)>
"""


from math import sqrt,fabs
import numpy,math
from numpy import *
from collections import defaultdict
from math import log


temp11=[]


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import sys,os




def calc_euclidean_distance(training_coordinates,query_coordinates):
    distance=sqrt(reduce(lambda x,y:x+y, map(lambda x,y:pow(x-y,2),training_coordinates,query_coordinates)))
    return distance


class myEM:
    """ my knn class"""
    def __init__(self,train_filename,K=3):

        #initialize simple parameters
        self.train_filename=train_filename
        #self.train_parsed_ip = self.parse_file(self.filereader(self.train_filename))
        self.K=K
        self.averages=[]
        self.variances=[]
        self.conditionalProbArray=[]
        self.weights = []


        


    def initParams(self,train_parsed_ip):
        features= array(map(lambda x:x[0],train_parsed_ip))
##        print "features 00000000", len(features)
        
        self.features = features
        self.labels = array(map(lambda x:x[1],train_parsed_ip))
        minmax=[]
        for i in range(0,len(features[0])):
            # the minimum and maximum value for each feature
            minmax.append((min(map(lambda x:x[i],features)),max(map(lambda x:x[i],features))))
        

        for i in range(0,self.K):
            #initialize averages so that min(feature)<= value <= max(feature)
            self.averages.append(map(lambda x:random.uniform(x[0],x[1]),minmax))

            #initialize covariance matrix to zeros
            var=zeros((len(features[0]),(len(features[0]))))
            
            temp_variances = map(lambda x: random.uniform(0,x),std(features,axis=0))

            #covariance matrix has diagonal elements being <= max variance in that feature
            for j in range(0,len(temp_variances)):
                var[j][j]=temp_variances[j]

            self.variances.append(array(var))


        self.averages=array(self.averages)
        #w array
        # pi array  [ [p11,p12... p1k], [p21,p22... p2k], ... upto n]
        self.weights = numpy.random.random(self.K)
        self.weights/=sum(self.weights)

        for i in range(0,len(features)):
            temp = numpy.random.random(self.K)
            temp/=sum(temp)
            try:
                self.conditionalProbArray.append(temp)
        
            except:
        
                pass
            

        self.conditionalProbArray = array(self.conditionalProbArray)
        
        
        

    def fxC(self, xi,k, det,inv):
        dx = xi - self.averages[k]
        m = len(self.averages[0])
        p = -0.5 * numpy.dot( numpy.dot(dx, inv), dx)
           
        b = (det*.5*numpy.pi)**(-m / 2.0)
        
        fxCVal = b * numpy.exp(p)
       # print "fxCVal----",fxCVal
        return fxCVal


    def iterateParameters(self):
        '''
        Iterates the parameters and return the log likelihood with the new
        parameter values
        '''
        
        newAverage = []
        newCov = []
        newWeights = []
        newConditionalProbs = []
        cond = self.conditionalProbArray.T
        updated_averages = []
        updated_weights = []
        updated_covs = []
        # update averages and weights
        for i in range(0,self.K):
            p = cond[i]
            tempval = zeros(len(self.features[0]))
            tempnum = zeros(len(self.features[0]))
            tempden =0
           # zeros(len(self.features[0]))
            for j in range(0,len(p)):
                t1 = p[j]
                t2 = self.features[j]
                t1t2 = t1*t2
                tempnum += t1t2
                tempden += p[j]
                
            tempval = tempnum/tempden
##            print "*****",len(cond[0])
            updated_weight = tempden/len(cond[0])

            
            updated_weights.append(updated_weight)
            updated_averages.append(tempval)

        self.averages = updated_averages
        self.weights = updated_weights


        for i in range(0,self.K):
            p = cond[i]
            tempval = zeros(len(self.features[0]))
            tempnum = zeros(len(self.features[0]))
            tempden = zeros(len(self.features[0]))
            for j in range(0,len(p)):
                t1 = p[j]

                #t2 = calc_euclidean_distance(self.features[j],self.averages[i])
                t2 = (self.features[j] - self.averages[i])**2
                t1t2 = t1*t2
                tempnum += t1t2
                tempden += p[j]
            tempval = tempnum/tempden
##            print "*****",len(cond[0])
            
            updated_covs.append(diag(tempval,0))

        self.variances = updated_covs

        fxc = zeros(shape(self.conditionalProbArray))

        
        # the expectation step
        for k in range(0,len(self.conditionalProbArray[0])):
            det = numpy.linalg.det(self.variances[k])

            try:
                inv = numpy.linalg.inv(self.variances[k])
            except:
                inv = det
            m = len(self.averages[0])
            a = det ** -0.5 * (2 * numpy.pi) ** (-m / 2.0)
            for i in range(0,len(self.conditionalProbArray)):
                
                fxCVal = self.fxC(self.features[i],k,det,inv)
                fxc[i][k] = fxCVal

##        print "tem ---------------------------",fxc
#        print sum(tem,axis =1)



        fx = []
        for i in range(0,len(fxc)):
            fxrow = 0
            for j in range(0,len(fxc[0])):
                fxrow+= self.weights[j]*fxc[i][j]
            fx.append(fxrow)


        tempConditional = zeros(shape(fxc))
        for i in range(0,len(tempConditional)):
            for j in range(0,len(tempConditional[0])):
                tempConditional[i][j] = (fxc[i][j]*self.weights[j])/fx[i]


        self.conditionalProbArray = tempConditional

        llharray = map(lambda x: log(x),fx)
        llh = reduce(lambda x,y:x+y,llharray)
        llh = -llh

##        print "llh in side",llh
        return llh


    def cluster(self):
        predictions = []
        for i in self.conditionalProbArray:
            j = list(i)
            predictions.append(j.index(max(j)))
        print "!!!!!!!!!!!!!predictions!!!!!!!!!!!!!",predictions

        return predictions


                
        
    
    def filereader(self,filename):
        """
        Reads a filename and returns a string of contents
        """
        f=open(filename)
        contents=f.read()
        f.close()
        return contents

    def parse_file(self,contents):
        """
        Accepts a string of contents and returns a structure of format [[[x1,x2,...xn],y],...]
        """
        holder=[]
        split_contents=contents.splitlines()
        #split_contents.append(
        temp=[]
        for i in split_contents:
            temp.append(i.split(','))
        del(split_contents)
        #numericData=nominaltoNumeric(temp,[4])
        numericData = temp
        #print "Numeric data", numericData

        #numericData = temp
        
        #comma_split_line=line.split(',')
        for line in numericData:
            #comma_split_line=line.split(',')
            ## because first column is row number
            #key=line[0:-1]
            key=map(lambda x: float(x),line[0:-1])
            value=line[-1]
            prep_list=[key,value]
            holder.append(prep_list)        
        return holder

    def test(self, testip):

        predictions = []
        for i in range(0,len(testip)):
            sc = self.cluster(testip[i])
            predictions.append((sc.index(max(sc)),testip[i][1]))
        return predictions
    

def crossValidate(train_parsed_ip1,N,x):
    '''
    should return the mean error for N folds
    '''
    slotsize=len(train_parsed_ip1)/N
    #print "Slot size-------------->",slotsize
    start=0
    llhs = []
    # Prepare Train and Test Sets
    for i in range(0,N):
        train_parsed_ip=[]
        end=start+slotsize
        test_parsed_ip = train_parsed_ip1[start:end]
        #print "len test ", len(test_parsed_ip)
        #print "test indices ",start, end
        if N!=1:
            for j in range(0,len(train_parsed_ip1)):
                if j not in range(start,end+1):
                    #print "Appended ",j,"to train"
                    train_parsed_ip.append(train_parsed_ip1[j])
        else:
            train_parsed_ip=test_parsed_ip
            
        start=end

        
        print "in cross validate",train_parsed_ip[0]
        
        res = driver(x,train_parsed_ip)
        pred = res[0]
        model = res[1]
        llh = res[2]
        llhs.append(llh)

    avg_llh = reduce(lambda x,y:x+y,llhs)
    avg_llh /=N

    return avg_llh
            
    
    

    
def driver(model,train_parsed_ip):
    


   # print ("len ----- trained_ parsed_ip", len(train_parsed_ip))

    x.initParams(train_parsed_ip)
    
    llh = -1

    while (True):
        llh1 = x.iterateParameters()
        if (llh1-llh)/llh < 0.00001:
            break
        else:
            llh = llh1
            continue

    print "Log likelihood------------>",llh1
        
    

    scores = []
    predictions = x.cluster()
    return [predictions,x,llh1]




if len(sys.argv)==2:
    k = int(sys.argv[1])
else:
    k=3
x=myEM('iris.data', 3)
parsed_data = x.parse_file(x.filereader(x.train_filename))
train_parsed_ip = parsed_data

p = driver(x,train_parsed_ip)

print "train_parsed_ip",train_parsed_ip[0]
#model
predictions = p[0]
model =p[1]
llh = p[2]
#plotting
xs=[]
ys=[]
zs=[]

fig = plt.figure()
ax = Axes3D(fig)

for i in model.features:
    xs.append(i[0])
    ys.append(i[1])
    zs.append(i[2])

for c, m, zl, zh in [('r', 'o', -50, -25),('b', '^', -30, -5)]:
    ax.scatter(xs, ys, zs, c=c, marker=m)

ax.set_xlabel('Sepal Length')
ax.set_ylabel('Sepal Width')
ax.set_zlabel('Petal Length')

plt.show()


## cross validation :
##llhs =[]
##for k in range(2,7):
##    x=myEM('/home/apurva/Desktop/sem3/DM/Assignment2/data/iris.data', k)
##    parsed_data = x.parse_file(x.filereader(x.train_filename))
##    train_parsed_ip = parsed_data
##    llh = crossValidate(train_parsed_ip,10,x)
##    llhs.append(llh)
##        
##
