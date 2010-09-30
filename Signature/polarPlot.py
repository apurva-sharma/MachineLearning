'''
Created on 12-Mar-2010

@author: Urjit Singh Bhatia
'''
import matplotlib.pyplot as plt

import csv, sys, os
fileToRead = sys.argv[1]

csvReader = csv.reader(open(fileToRead))
forg = []
org = []
forgi = []
orgi = []
i=0
orgAngle = 0
forgAngle = 180
for row in csvReader:
	if(i>0):
		forg.append(row)
		forgi.append(i-1)
		orgAngle += 1
		if(orgAngle>180):
			orgAngle = 0
	if(i>=31682):
		org.append(row)
		orgi.append(i-1)
		forgAngle += 1
		if(forgAngle>360):
			forgAngle = 180
	i = i + 1

plt.title('Genuines versus Forgeries in the distance space')
plt.polar(forgi, forg,'bo')
plt.polar(orgi,org,'ro')
plt.show()