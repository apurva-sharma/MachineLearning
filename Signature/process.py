'''
Created on 12-Mar-2010

@author: Urjit Singh Bhatia
'''
import csv, sys, os, math
fileToRead = sys.argv[1]
forgFile = sys.argv[2]

doOrig = sys.argv[3]

csvReader = csv.reader(open(fileToRead))
csvForgReader = csv.reader(open(forgFile))


diff = ""
countr = 0
countc = 0
tempFlag = 0

data = []
temprow = []

for row in csvReader:
	length = len(row)
	temprow = []
	for i in range(0,length):
		temprow.append(row[i])
	row=map(lambda x: float(x),row)
	data.append(row)

#print data
#print len(data)

if(int(doOrig) == 0):
	count = int(0)
	tcount = 0

	file = open("orgdiffForgTest.csv","wb")
	
	dataf = []
	for row in csvForgReader:
		length = len(row)
		temprow = []
		for i in range(0,length):
			temprow.append(row[i])
		row=map(lambda x: float(x),row)
		dataf.append(row)

	#print dataf
	#print len(dataf)

	while((count + 24) <= len(data)):
		for rowI in range(count,int(count+24)):
			for rowJ in range(count,int(count+24)):
				#if(rowI != rowJ):
				diff = ""
				for col in range(0,length):
					diff += str(abs(float(data[rowI][col]) - float(dataf[rowJ][col])))
					diff += ","
				#print diff
				#sys.exit()
				diff += "1"
				file.write(diff)
				file.write("\n")
		count += 24
		tcount += 1
		print count
else:
	file = open("orgdiffOrgTest.csv","wb")
	count = 0
	tcount = 0
	while((count + 24) <= len(data)):
		for rowI in range(count,int(count+24)):
			for rowJ in range(rowI+1,int(count+24)):
				#if(rowI != rowJ):
					diff = ""
					for col in range(0,length):
						diff += str(abs(float(data[rowI][col]) - float(data[rowJ][col])))
						diff += ","
					#print rowI
					#sys.exit()
					diff += "0"
					file.write(diff)
					file.write("\n")
		count += 24
		tcount += 1
		print count