#include <cv.h>
#include <highgui.h>
#include "ChainCodeExtractor.h"


using namespace chainCode;
using namespace cv;
using namespace std;

//simple constructor
ChainCodeExtractor::ChainCodeExtractor()
{
}

void ChainCodeExtractor::initChainCodeReader(IplImage* imgSrc, int retrievalMethod)
{
	this->img = imgSrc;
	this->retrievalMethod = retrievalMethod;
	this->total = getTotalContours(retrievalMethod);
}

int ChainCodeExtractor::getTotalContours(int retrievalMethod)
{
	CvMemStorage* memory = cvCreateMemStorage();
	CvSeq* temp = NULL;
	int total = cvFindContours(this->img, memory, &temp, sizeof(CvChain), retrievalMethod, CV_CHAIN_CODE);
	return total;
}

int** ChainCodeExtractor::getNextChainCodes(bool supressDebug)
{
	int contourCounter = 0;
	CvSeq* _contour;
	CvMemStorage* memory = cvCreateMemStorage();
	CvSeqReader reader;
	CvContourScanner scanner =  cvStartFindContours(this->img, memory, sizeof(CvChain), this->retrievalMethod, CV_CHAIN_CODE);

	bool exhausted = true;
	//int array to hold the chain codes.
	
	int** currentContourCodes = (int**)malloc(sizeof(int*) * this->total);
	//currentContourCodes = new int*[this->total];

	/* Retrieves next contour */
	_contour =  cvFindNextContour(scanner );
	int lengthOfChain, chainCounter;
	char code;

	while(_contour)
	{
		cvStartReadSeq (_contour, &reader );
		lengthOfChain = _contour->total;
		
		if(lengthOfChain < 1)
		{
			_contour = cvFindNextContour( scanner );	
			continue;
		}

		currentContourCodes[contourCounter] = new int[lengthOfChain];

		for ( chainCounter = 0 ; chainCounter < lengthOfChain ; chainCounter++)
		{
			CV_READ_SEQ_ELEM ( code, reader );

			*(*(currentContourCodes + contourCounter) + chainCounter) = code;
		}

		*(*(currentContourCodes + contourCounter) + chainCounter) = -1;

		
		_contour = cvFindNextContour( scanner );

		if(!supressDebug)
			printf("\n count is %d",contourCounter);
		
		contourCounter++;
	}

	*(currentContourCodes + contourCounter) = NULL;
	
	return currentContourCodes;
}
