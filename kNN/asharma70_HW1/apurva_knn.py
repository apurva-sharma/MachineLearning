#!/usr/bin/env python
"""
Author :Apurva Sharma
email  :asharma70@gatech.edu
GTID   :902490301
Simple knn algorithm
usage : python myknn.py <train file> <test file> <k- optional (default 4)>
"""


from math import sqrt,fabs
from numpy import *


temp11=[]

import sys,os
class MyKnn:
    """ my knn class"""
    def __init__(self,train_filename,k=3):
        self.train_filename=train_filename
        self.k=k
    
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
        split_contents=contents.splitlines()[1:]
        #split_contents.append(
        temp=[]
        for i in split_contents:
            temp.append(i.split(','))
        del(split_contents)
        numericData=nominaltoNumeric(temp,[5])

        
        
        #comma_split_line=line.split(',')
        for line in numericData:
            #comma_split_line=line.split(',')
            ## because first column is row number
            key=line[1:-1]
            key=map(lambda x: float(x),key)
            value=float(line[-1])
            prep_list=[key,value]
            holder.append(prep_list)        
        return holder
    
            
            
        

    def calc_distance(self,training_point,query_point,fn):
        """
        accepts a single training point [[x1,x2,...xn],y] and query point [[x1,x2,...xn],y]
        returns a dictionary of form {index:distance}
        """
        training_coordinates=training_point[0]
        query_coordinates=query_point[0]
        distance=fn(training_coordinates,query_coordinates)
        return distance
    
    
    def find_nearest_neighbours(self,distance_dictionary):
        """
        accepts distance_dictionary 
        returns a list containing indices of k nearest neighbours
        """
        d=distance_dictionary.values()
        d.sort()
        bag_of_indices=[]
        closest_distances=d[0:self.k]
        for i in closest_distances:
            for key in distance_dictionary.keys():
                if distance_dictionary[key]==i:
                    bag_of_indices.append(key)
        return bag_of_indices

    def predict_value(self,neighbour_list):
        """
        list of form [[x1,x2,...xn],y] containing k such neighbours
        returns 1 averaged prediciton value
        """
        prediction = reduce(lambda x,y:x+y,map(lambda x:x[1],neighbour_list))/len(neighbour_list)
        if prediction>0.5:
            return 1
        else:
            return 0
        #return prediction



def nominaltoNumeric(contents,indices):
        for i in indices:
            unique_labels=list(set(map(lambda x:x[i],contents)))
            label_dic={}
            for j in range(0,len(unique_labels)):
                label_dic[unique_labels[j]]=j
            for j in range(0,len(contents)):
                contents[j][i] = label_dic[contents[j][i]]

            return contents
        
def calc_euclidean_distance(training_coordinates,query_coordinates):
        distance=sqrt(reduce(lambda x,y:x+y, map(lambda x,y:pow(x-y,2),training_coordinates,query_coordinates)))
        return distance
    
def calc_hamming_distance(training_coordinates,query_coordinates):
        zipped_distances=zip(training_coordinates,query_coordinates)
        distance=sum(c1!=c2 for c1,c2 in zipped_distances)
        weight=1.0/distance
        weighted_distance=distance*weight
        return distance
    
def calc_manhattan_distance(training_coordinates,query_coordinates):
        distance=(reduce(lambda x,y:x+y, map(lambda x,y:abs(x-y),training_coordinates,query_coordinates)))
        return distance
    
def calc_inverse_distance_weight(training_coordinates,query_coordinates):
        distance=sqrt(reduce(lambda x,y:x+y, map(lambda x,y:pow(x-y,2),training_coordinates,query_coordinates)))
        weight=1.0/distance
        weighted_distance=distance*weight
        return weighted_distance

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
    
    


def dokNN(train_parsed_ip,test_parsed_ip,x):
    '''
    returns predicted values for test set
    '''
    """
    distance_dic holds {index:distance}
    """
    distance_dic={}
    prediction_holder=[]
    """
    Calculate distances for each test point with all the training points
    """

    for j in range(0,len(test_parsed_ip)):
    ## For each entry in test ip                
        for k in range(0,len(train_parsed_ip)):
            d=x.calc_distance(train_parsed_ip[k],test_parsed_ip[j],calc_euclidean_distance)
            distance_dic[k]=d
        indices_of_neighbours=x.find_nearest_neighbours(distance_dic)
        populated_neighbours=[]
        """
        populate the neigbours
        """
        for index in indices_of_neighbours:
            populated_neighbours.append(train_parsed_ip[index])
        predicted_value=x.predict_value(populated_neighbours)
#        print "*********predicted_value***** ",predicted_value
        prediction_holder.append(predicted_value)

    return prediction_holder




    
    """
    Calculate mean error
    """

"""
Driver
"""

    
if len(sys.argv)==4:
    x=MyKnn(sys.argv[1],int(sys.argv[2]))
    N= int(sys.argv[3])

else:
    print "Correct usage : python myknn.py <train file> <k> <N (folds)>"
    sys.exit()
    
    

#x=MyKnn(sys.argv[1],int(sys.argv[2]))
    


train_contents=x.filereader(x.train_filename)
train_parsed_ip1=x.parse_file(train_contents)


train_parsed_ip1=normalize_data(train_parsed_ip1)

#test_contents=x.filereader(x.test_filename)
#test_parsed_ip=x.parse_file(test_contents)
#test_parsed_ip=normalize_data(test_parsed_ip)


#print test_parsed_ip[0]


#predictions= test(train_parsed_ip1,N,x)

meanErrorArray = crossValidate(train_parsed_ip1,N,x)
print "Errors Accross Folds---->\n",meanErrorArray

meanError = reduce(lambda x,y:x+y,meanErrorArray)/float(N)
print "Mean Error Accross N Folds----> ",meanError



finalPredictions = dokNN(train_parsed_ip1,train_parsed_ip1,x)


op_string=''
op_string+='Actual Value,Prediction\n'
numWrongs=0
for i in range(0,len(finalPredictions)):
               op_string+=str(train_parsed_ip1[i][1])+','+ str(finalPredictions[i])+'\n'
               if int(train_parsed_ip1[i][1])<>int(finalPredictions[i]):
                   numWrongs+=1
               
f=open('op.csv','w')
f.write(op_string)
f.close()
print "Output generated in >>> ",os.getcwd(),"/op.csv"
print "Number of Wrong Predictions on when Trained and Tested on Full Dataset ", numWrongs
print "% wrong ", (float(numWrongs)/(len(train_parsed_ip1)))*100

