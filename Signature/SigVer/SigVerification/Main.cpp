#include <cv.h>
#include <highgui.h>
#include "GradientHelper.h"
#include "MeasureOfPenPressure.h"
#include "PreProcess.h"

using namespace cv;
using namespace gsc;
using namespace mopp;
using namespace prep;


int main(int argc, char** argv)
{

	const char* imagename = argc > 1 ? argv[1] : "original_2_18.png";

	
	//load the image in grayscale
	IplImage* img = cvLoadImage(imagename);
	printf("asdasda");
	MeasureOfPenPressure mopp;
	float answer = mopp.getEntropy(img);
	double k = mopp.getGrayLevelThreshold(img);
	cvThreshold(img,img, k, 255, CV_THRESH_BINARY);
	double countBlack = mopp.getNumberOfBlackPixels(k,img);

	PreProcess processor;
	img = processor.preprocessing(img,50,50);

	cvShowImage("hello",img);

	IplImage* temp = cvLoadImage(imagename,0);
	cvThreshold(temp,temp, k, 255, CV_THRESH_BINARY_INV);
	//temp = processor.preprocessing(temp,50,50);

	CvSeq *_contour;
	CvMemStorage* memory = cvCreateMemStorage();
	//cvFindContours(temp, memory, &_contour, sizeof(CvChain),CV_RETR_LIST, CV_CHAIN_CODE);
	CvContourScanner scanner =  cvStartFindContours( temp, memory,sizeof(CvChain),CV_RETR_LIST, CV_CHAIN_CODE);

	/* Retrieves next contour */
	_contour =  cvFindNextContour( scanner );
	CvSeqReader reader;
	
	CvScalar red = CV_RGB(250,0,0);
    CvScalar blue = CV_RGB(0,0,250);
	IplImage* img3 = cvLoadImage(imagename);
	cvThreshold(img3,img3, k, 255, CV_THRESH_BINARY_INV);
	//img3 = processor.preprocessing(img3,50,50);
	cvShowImage("temp",img3);

	int count = 0;
	while(_contour)
	{
		
		cvStartReadSeq ( _contour, &reader );
		int num = _contour->total;

		char code;
		int sum = 0;

		for ( int i = 0 ; i < num ; ++i )
		{
			CV_READ_SEQ_ELEM ( code, reader );

			if(code == 0)
				printf("0");

			else if(code == 1)
				printf("1");

			else  if(code == 2)
				printf("2");

			else  if(code == 3)
				printf("3");

			else  if(code == 4)
				printf("4");

			else  if(code == 5)
				printf("5");

			else  if(code == 6)
				printf("6");

			else if(code == 7)
				printf("7");

			else
				printf("no");


		}

		cvDrawContours(
			img3,
			_contour,
			red,		// Red
			blue,		// Blue
			1,			// Vary max_level and compare results
			2,
			8 );

		cvShowImage( "Contours 2", img3 );
		cvWaitKey(100);
		printf("\n*********\n");
		//printf("*\n");
		_contour =  cvFindNextContour( scanner );
		count++;
	}
	printf("\ncount is %d",count);



	/*
	//CvChainPtReader* reader;
	CvMemStorage* memory = cvCreateMemStorage();
	CvSeq* first_contour = NULL;
	CvChainPtReader crdReader;
	CvPoint pntPoint;
	
	//cvFindContours(temp,memory,&first_contour,,,CV_CHAIN_CODE,);
	cvFindContours(temp, memory, &first_contour, sizeof(CvChain),CV_RETR_LIST, CV_CHAIN_CODE);
	
	int iContourLength = first_contour->total;

	//cvStartReadChainPoints((CvChain*) first_contour, &crdReader);

	CvSeqReader reader;
	cvStartReadSeq ( first_contour, &reader );

	for(int i = 0; i < iContourLength; i++)
	{
		char code;
		CV_READ_SEQ_ELEM ( code, reader );
		if(code == '0')
			printf("hello");
	}



	CvSeq* result;

	while( first_contour )
    {
		result = cvApproxPoly( first_contour, sizeof(CvContour), memory,
			CV_POLY_APPROX_DP, cvContourPerimeter(first_contour)*0.02, 0 );
		

		for( int i=0; i<first_contour->total; ++i ) 
		{
			CvPoint* p = (CvPoint*)cvGetSeqElem(first_contour,i);
				printf("hello x is %d\t",p->x);
				printf("hello y is %d\n",p->x);
		}
		first_contour = first_contour->h_next;
	}
	for( CvSeq* c = first_contour; c!=NULL; c=c->h_next ) 
	{
		for( int i=0; i<c->total; ++i ) 
		{
			/**
			* Code to put here to display Freeman chain code
			* representation of each contour
			*/
			/*
			CvSeqBlock* block = c->first;
			for(int h=0;h<block->count;h++)
			printf("%d",block->data[h]);

			printf("\n");
			while((block = block->next))
			for(int h=0;h<block->count;h++)
			printf("%d",block->data[h]);
			printf("\n");
			*/
		/*
			CvPoint* p = (CvPoint*)cvGetSeqElem(c,i);
			printf("x is %d\t",p->x);
			printf("y is %d\n",p->x);
		}
	}
	*/


	//GradientHelper gradientHelper;
	//gradientHelper.getGradientMatrix(img, &sobleDest);

	//cvDestroyAllWindows();
	//return(0);
	/*
	Point2f src_center(img->width/2.0, img->height/2.0);
	Mat rot_mat = getRotationMatrix2D(src_center, 90, 1.0);
	Mat dst;
	Mat imgMat = cvarrToMat(img,true,false,0);
	warpAffine(imgMat, dst, rot_mat, imgMat.size());

	IplImage img_to_show = dst;
	//img_to_show.align = 

	cvNamedWindow(imagename,1);
	cvShowImage("rotated",&img_to_show);
	GradientHelper gh;
	cvWaitKey(0);

	return 1;
	printf("%d",img->height);

	CvRect bb;//bounding box

	CvMat data;

	bb = findBB(img);

	cvGetSubRect(img, &data, bb);

	cvShowImage(imagename, &data);
	cvWaitKey(0);
	getchar();
	printf("%d",data.height);

	img = preprocessing(img, 500, 500);
	if(!img)
	return -1;
	cvShowImage(imagename, img);


	getchar();
	*/
	cvWaitKey(0);
}
