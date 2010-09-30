'''
	ML script
'''

import sys,os, subprocess, win32api,glob

long_file_name = "C:\\Users\\admin\\Documents\\Visual Studio 2008\\Projects\\SigVer\\Debug\\SigVerification.exe"
short_file_name = win32api.GetShortPathName(long_file_name)

try:
	outputfilename = sys.argv[2]
	inputfolder = sys.argv[1]
	fileextension = sys.argv[3]
	
	inputfolder = inputfolder.replace("\\","\\\\")
	print "Reading images from ",inputfolder," with extension ",fileextension
	print "Writing output to ",outputfilename
except:
	print "outputfilename = sys.argv[2], inputfolder = sys.argv[1]"
	sys.exit(0)

print short_file_name

#file = open("c:\\signatureml\\PfilenamesForg.csv",'wb')

#for infile in glob.iglob(os.path.join("C:\\signatureml\\signatures\\full_org\\","*.png")):
for infile in glob.iglob(os.path.join(inputfolder,"*." + fileextension)):
	command = short_file_name + " " + infile
	p = os.system(command + " " +str(outputfilename))
	#print command
	#print p
	print "processing... ",infile
	#sys.exit(0)
	#file.write(infile + "\n")
	