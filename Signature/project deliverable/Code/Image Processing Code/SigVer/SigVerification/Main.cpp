#include <cv.h>
#include <highgui.h>
#include <math.h>
#include <windows.h>
#include <tchar.h>

#include "GradientHelper.h"
#include "MeasureOfPenPressure.h"
#include "PreProcess.h"
#include "ChainCodeExtractor.h"

using namespace cv;
using namespace gsc;
using namespace mopp;
using namespace prep;
using namespace chainCode;


int main(int argc, char** argv)
{
	/*
	LPWIN32_FIND_DATAW imageHandle;
	HANDLE find;
	syste
	LPCWSTR dir = L"C:/mlas";
	find = FindFirstFile(dir, imageHandle);
	if (find == INVALID_HANDLE_VALUE) 
	{
		printf ("FindFirstFile failed %d\n", GetLastError());
		return 0;
	} 
	else 
	{
		printf("The first file found is %s\n",imageHandle->cFileName);
		FindClose(imageHandle);
	}


	//if(imageHandle)
	//	printf("dir %s", imageHandle->cFileName);
	//char* filename;
	//wctomb(filename,imageHandle->cFileName);
	*/
	
	FILE* outFile = freopen(argv[2],"a",stdout);

	if(outFile == NULL)
	{
		getchar();
	}
	//HANDLE hfile = CreateFile(L"C:\mlas\output.txt",GENERIC_READ,0x00000001,NULL,OPEN_ALWAYS,FILE_ATTRIBUTE_NORMAL,FILE_ATTRIBUTE_NORMAL);
	

	const char* imagename = argc > 1 ? argv[1] : "original_2_18.png";

	
	//load the image in grayscale
	IplImage* img = cvLoadImage(imagename);
	
	//mopp
		MeasureOfPenPressure mopp;
		float answer = mopp.getEntropy(img);
		double k = mopp.getGrayLevelThreshold(img);
		int countBlack = mopp.getNumberOfBlackPixels(k,img);

		printf("%f,",answer);
		printf("%f,",k);
		printf("%d,",countBlack);
	//end-mopp

	//Preprocess
		PreProcess processor;

		cvThreshold(img,img, k, 255, CV_THRESH_BINARY);
		img = processor.preprocessing(img,50,50);
		//cvShowImage("hello",img);
	//end-Preprocess

	//chaincodes
	//check why different loading styles work
	img = cvLoadImage(imagename,0);
	cvThreshold(img,img, k, 255, CV_THRESH_BINARY);
	img = processor.preprocessing(img,50,50);
	cvThreshold(img,img, k, 255, CV_THRESH_BINARY_INV);

	/*
	//Print if print flag is set
	const char*	newName = argv[2];

	if(!cvSaveImage(newName,img))
		printf("Could not save: %s\n",newName);

	exit(0);
	*/

	ChainCodeExtractor chainCoder;
	chainCoder.initChainCodeReader(img,CV_RETR_LIST);
	int exterior = chainCoder.getTotalContours(CV_RETR_EXTERNAL);
	int interior = chainCoder.getTotalContours(CV_RETR_LIST) - exterior;

	//printf("\nexteriors %d",exterior);
	//printf("interiors %d\n",interior);

	printf("%d,",interior);
	printf("%d,",exterior);

	int** chains, contourCounter = 0, chainCounter = 0;
	
	chains = chainCoder.getNextChainCodes(true);
	int code;
	
	float numVerticalContours = 0, numHorizontalContours = 0,  numPositiveContours = 0,  numNegativeContours = 0;
	float n1,n3;
	n1 = n3 = 0;

	//n2 = nVertical

	while(*(chains + contourCounter) != NULL)
	{
		chainCounter = 0;
		while( *(*(chains + contourCounter) + chainCounter) != -1)
		{
			code = *(*(chains + contourCounter) + chainCounter);
			//printf("%d", code);

			//count verticals
			if(code == 2 || code == 6)
				numVerticalContours++;

			//count horizontals
			if(code == 0 || code == 4)
				numHorizontalContours++;

			//count positives
			if((code >= 0 && code <= 2) || (code == 5))
				numPositiveContours++;

			//count negatives
			if(code == 3 || code == 4 || code == 6 || code == 7)
				numNegativeContours++;

			if(code == 1 || code == 5 )
				n1++;

			if(code == 3 || code == 7)
				n3++;

			chainCounter++;
		}
		contourCounter++;
		//printf("\n************************\n");
	}

	float slant = atanf((float) (n1 - n3)/(n1 + numVerticalContours + n3));

	/*
	printf("\n vertical %d\n", numVerticalContours);
	printf("\n horizontals %d\n", numHorizontalContours);

	printf("\n positives %d\n", numPositiveContours);
	printf("\n negatives %d\n", numNegativeContours);

	
	printf("\n slant %f\n",slant);

	printf("\n height %d\n",img->height);
	*/

	float sum = numVerticalContours + numHorizontalContours + numPositiveContours + numNegativeContours;
	
	numVerticalContours /= sum;
	numHorizontalContours /= sum;
	numPositiveContours /= sum;
	numNegativeContours /= sum;

	printf("%f,",numVerticalContours);
	printf("%f,",numHorizontalContours);
	printf("%f,",numPositiveContours);
	printf("%f,",numNegativeContours);

	printf("%f,",slant);
	printf("%d\n",img->height);



	//cvWaitKey(0);
}
