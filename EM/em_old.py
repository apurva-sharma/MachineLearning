#!/usr/bin/env python
"""
Author :Apurva Sharma
email  :asharma70@gatech.edu
GTID   :902490301
Simple knn algorithm
usage : python em.py <train file> <test file> <k- optional (default 4)>
"""


from math import sqrt,fabs
import numpy,math
from numpy import *
from collections import defaultdict
from math import log
temp11=[]



import sys,os
class myEM:
    """ my knn class"""
    def __init__(self,train_filename,K=3):

        #initialize simple parameters



        
        self.train_filename=train_filename
        self.train_parsed_ip = self.parse_file(self.filereader(self.train_filename))
        self.K=K
        self.averages=[]
        self.variances=[]
        
        features= array(map(lambda x:x[0],self.train_parsed_ip))
        self.features = features
        self.labels = array(map(lambda x:x[1],self.train_parsed_ip))
        minmax=[]
        for i in range(0,len(features[0])):
            # the minimum and maximum value for each feature
            minmax.append((min(map(lambda x:x[i],features)),max(map(lambda x:x[i],features))))
            #mins.append(min(map(lambda x:x[i],features)))
            #maxes.append(max(map(lambda x:x[i],features)))

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
        self.conditionalProbArray=[]
        # pi array  [ [p11,p12... p1k], [p21,p22... p2k], ... upto n]
        self.weights = numpy.random.random(self.K)
        self.weights/=sum(self.weights)

        for i in range(0,len(features)):
            temp = numpy.random.random(self.K)
            temp/=sum(temp)
            self.conditionalProbArray.append(temp)
            

        self.conditionalProbArray = array(self.conditionalProbArray)
        
        
        #self.variances=[]

    def fxC(self, xi,k, det,inv):
        dx = xi - self.averages[k]
        m = len(self.averages[0])
        p = -0.5 * numpy.dot( numpy.dot(dx, inv), dx)
        b = (det*.5*numpy.pi)**(-m / 2.0)
        fxCVal = b * numpy.exp(p)    
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

##        print "updated_averages",updated_averages
##        print "updated_wts",updated_weights
        self.averages = updated_averages
        self.weights = updated_weights


        for i in range(0,self.K):
            p = cond[i]
            tempval = zeros(len(self.features[0]))
            tempnum = zeros(len(self.features[0]))
            tempden = zeros(len(self.features[0]))
            for j in range(0,len(p)):
                t1 = p[j]
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
        for k in range(0,len(x.conditionalProbArray[0])):
            det = numpy.linalg.det(self.variances[k])
            inv = numpy.linalg.inv(self.variances[k])
            m = len(self.averages[0])
            a = det ** -0.5 * (2 * numpy.pi) ** (-m / 2.0)
            for i in range(0,len(x.conditionalProbArray)):
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

##        print "fx", fx
##        print "len fx---------" , len(fx) 
##        print "wts----------------\n",self.weights

        tempConditional = zeros(shape(fxc))
        for i in range(0,len(tempConditional)):
            for j in range(0,len(tempConditional[0])):
                tempConditional[i][j] = (fxc[i][j]*self.weights[j])/fx[i]


        self.conditionalProbArray = tempConditional

        llharray = map(lambda x: log(x),fx)
        llh = reduce(lambda x,y:x+y,llharray)
        llh = -llh

        
        return llh


    def cluster(self,point):
        scores = []
        for i in range(0,self.K):
            scores.append(self.calc_score(self.averages[i],self.weights[i],self.variances[i], point))
        return scores

    def gaussian(self,averages,variances,point):
        det = numpy.linalg.det(variances)
        inv = numpy.linalg.inv(variances)
        m = len(point)
        a = det ** -0.5 * (2 * numpy.pi) ** (-m / 2.0) 
        dx = point - averages
        print dx, inv
        b = -0.5 * numpy.dot( numpy.dot(dx, inv), dx)
        print "gaussain returns************** ",a * numpy.exp(b) 
        return a * numpy.exp(b) 
        

    def calc_score(self,averages,weight,variances,point):
        llh = weight * self.gaussian(averages,variances,point)
        return llh
            


##        
##        # getting the gaussian
##        m = len(self.averages[0])
##        det = numpy.linalg.det(self.variances[0])
##        inv = numpy.linalg.inv(self.variances[0])
##        a = det ** -0.5 * (2 * numpy.pi) ** (-m / 2.0)
##        x = self.features[0]
##        mean = self.averages[0]
##        dx = x - mean
##        print dx, inv
##        b = -0.5 * numpy.dot( numpy.dot(dx, inv), dx)
##        print "gaussain returns************** ",a
##        print "***********",b

##        m = len(self.averages[0])
##        a = det ** -0.5 * (2 * numpy.pi) ** (-m / 2.0)
##        print "A",a
##        dx = self.features[0] - self.averages[0]
##        print "dx----------------",dx,'\n'
##        self.dx = dx
##        
##        #print "dx", dx
##        
##        #print dx, inv
##
##        
##        #b = numpy.dot(dx, dx)
##        #print "b",b
##
##        #c = a * numpy.exp(b)
##        #print "c", c
##        
        
        
        
        

        


        ## calculate sigma
        
##            
##            print "shape p",shape(p)
##            print "shape features T",shape(self.features[i].T)
##            print "features [i]",self.features[i].T
##            print p * self.features[i].T
##            print "****************"
        
            
            #self.features
            
                
        
    
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
        print "Numeric data", numericData

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
    
            
            
        

    

    
def mapPredictionstoLabels(predictions):
    tuples = predictions
##    for i in range(0,len(predictions)):
##        tuples.append((predictions[i],labels[i]))

    for i in tuples:
        print "tuple entry ", i,"added"

    freq_dict = defaultdict(int)

    for i in tuples:
        freq_dict[i]+=1

    for i,j in freq_dict.items():
        print i, '**', j

    sorted_tuples = sorted([(value,key) for (key,value) in freq_dict.items()],reverse = True)[0:3]
    sorted_tuples = map(lambda x: x[1],sorted_tuples)
    
    return sorted_tuples
    
        
def nominaltoNumeric(contents,indices):
    print indices
    print contents
    for i in indices:
        unique_labels=list(set(map(lambda x:x[i],contents)))
        label_dic={}
        for j in range(0,len(unique_labels)):
            label_dic[unique_labels[j]]=j
        for j in range(0,len(contents)):
            contents[j][i] = label_dic[contents[j][i]]

        return contents
    


def normalize_data(bag):
    features=map(lambda x:x[0],bag)
    outputs=map(lambda x:x[1],bag)
    mins=[]
    maxes=[]
    #means=mean(features,axis=0)
    #stds=std(features,axis=0)
    for i in range(0,len(features[0])):
        mins.append(min(map(lambda x:x[i],features)))
        maxes.append(max(map(lambda x:x[i],features)))
    

    for i in range(0,len(features)):
        for j in range(0,len(features[0])):
            features[i][j]=(features[i][j]-mins[j])/maxes[j]

    normalized_data=[]

    #print "************ mins********\n",mins
    map(lambda x,y:normalized_data.append([x,y]),features,outputs)
    return normalized_data




def crossValidate(train_parsed_ip1,N,x):
    '''
    should return the mean error for N folds
    '''
    slotsize=len(train_parsed_ip1)/N
    #print "Slot size-------------->",slotsize
    start=0
    meanErrorArray=[]
    # Prepare Train and Test Sets
    for i in range(0,N):
        numwrongs=0
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
        prediction_holder = dokNN(train_parsed_ip,test_parsed_ip,x)
        
        #output_string=''

        actual_value_list=[]

        for i in range(0,len(test_parsed_ip)):
            actual_value_list.append(test_parsed_ip[i][1])
            numwrongs+= int(test_parsed_ip[i][1])^prediction_holder[i]
            #output_string+=str(prediction_holder[i])+','+str(test_parsed_ip[i][1])+"\n"

        error= reduce(lambda x,y:x+y, map(lambda x,y:fabs(x-y),prediction_holder,actual_value_list))
        error/=len(actual_value_list)
        meanErrorArray.append(numwrongs)
        #meanErrorArray.append(error)
        #print meanErrorArray

    return meanErrorArray
    
    




    
    """
    Calculate mean error
    """

"""
Driver
"""

####    
####if len(sys.argv)==4:
####    x=MyKnn(sys.argv[1],int(sys.argv[2]))
####    N= int(sys.argv[3])
####
####else:
####    print "Correct usage : python myknn.py <train file> <k> <N (folds)>"
####    sys.exit()
####    
    
def driver():
    x=myEM('/home/apurva/Desktop/sem3/DM/Assignment2/data/iris.data', 3)
    llh = -1

    while (True):
        llh1 = x.iterateParameters()
        print "----llh1---",llh1
        print "----",(llh1-llh)/llh
        if (llh1-llh)/llh < 0.1:
            break
        else:
            llh = llh1
            continue
        
    

    scores = []
    predictions = []
    for i in range(0,len(x.features)):
        sc = x.cluster(x.features[i])
        predictions.append((sc.index(max(sc)),x.labels[i]))

    sorted_tuples = mapPredictionstoLabels(predictions)
    return [sorted_tuples,predictions,x]
    



def driver1():
    clusternames = []
    while(True):
        results = driver()
        
        sorted_tuples = results[0]
        predictions = results[1]
        model = results[2]
        clusternames = set(map(lambda x:x[1],sorted_tuples))
        clusternumbers = set(map(lambda x:x[0],sorted_tuples))

        if (len(clusternames)==3 and len(clusternumbers) == 3):
            break

    return [sorted_tuples,predictions,x]

    
        
    
if __name__=='__main__':
    results = driver1()
    sorted_tuples = results[0]
    predictions = results[1]
    x = results[2]

    
    error = 0 
    for i in predictions:
        if i not in sorted_tuples:
            error+=1
    print error
