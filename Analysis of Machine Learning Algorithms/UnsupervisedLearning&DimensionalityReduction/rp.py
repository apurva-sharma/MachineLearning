import numpy
from numpy import *
import math
import sys
from scipy import linalg, mat, dot


#ip=array([[1,1,0,0],[1,0,1,0],[1,1,1,1],[0,0,0,0],[1,0,0,1]])


def reader(filename):
    f=open(filename)
    contents=f.readlines()
    f.close()
    parsed_contents= map(lambda x:x.split(','),map(lambda x:x.strip(),contents))
    for i in range(0,len(parsed_contents)):
        parsed_contents[i]=map(lambda x:float(x),parsed_contents[i])
    return array(parsed_contents)
        
    

def RandomizedProjectionFilter(componentsOut,componentsIn):
    projections=random.random((componentsOut,componentsIn))
    temp=ones((componentsOut,componentsIn))*.5
    projections=projections-temp
    U, s, V = linalg.svd(projections)
    if(componentsIn<=componentsOut):
        projections=U[0:projections.shape[0],0:projections.shape[1]]
    else:
        projections=V[0:projections.shape[0],0:projections.shape[1]]
    return projections


def times(projections,vec):
    res= zeros((1,projections.shape[0]))
    for i in range(0,shape(projections)[0]):
        for j in range(0,shape(projections)[1]):
            #print "projections[i][j]*vec[j]",projections[i][j]*vec[j]
            res[0][i]+=projections[i][j]*vec[j]
    return res


def myfilter(ds,prj):
    bag=[]
    for i in ds:
        bag.append(times(prj,i))
    return bag
	    

def myreverse(ds,prj):
    bag=[]
    for i in ds:
        bag.append(times(prj,i))
    return bag
        

def parse_filters(unparsed_filter):
    parsed_filter=[]
    for i in unparsed_filter:
        for j in i:
            parsed_filter.append(list(j))
    parsed_filter=array(parsed_filter)
    return parsed_filter

def calc_error(ip,recovered):
    #print "ip--",ip[0]
    #print "recovered--",recovered[0]
    diff_bag=abs(ip-recovered)
    #print "diff--\n", diff_bag[0]
    diff_bag=map(lambda x:list(x),list(diff_bag))
    error=0
    for i in diff_bag:
        #error+=reduce(lambda x,y:x**2+y**2,i)
        error+=reduce(lambda x,y:abs(x)+abs(y),i)
    return log(error)
    
    
bag=[]
error_bag=[]


#print sys.argv[1]
ip=reader(sys.argv[1])
#iterations=int(sys.argv[2])
iterations=100
print "ip prepared"
for i in range(0,iterations):
    ## projection multiplier matrix
    myprojections=RandomizedProjectionFilter(10,58)

    ## filtered projections
    filtered=myfilter(ip,myprojections)
    parsed_filter=parse_filters(filtered)
   # print "$$$$parsed_filter" , parsed_filter


    ## recovered projections
    recovered=myreverse(parsed_filter,transpose(myprojections))
    parsed_recover=parse_filters(recovered)
    #diff_bag=abs(parsed_recover-ip)
    error_bag.append(calc_error(ip,parsed_recover))
    #print "diff bag"
    #print diff_bag
    bag.append(parsed_recover)

#print "Input \n", ip

##print "Projections \n",parsed_filter
##
##print "Recovered \n",parsed_recover

#print bag

#mean_bag=mean(bag,axis=0)
#print " Mean Recovery\n "
#print mean_bag

print "error bag\n",error_bag


