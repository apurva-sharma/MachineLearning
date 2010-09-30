#include "PreProcess.h"
#include <cv.h>
#include <highgui.h>

using namespace std;
using namespace prep;

PreProcess::PreProcess()
{
}
CvRect PreProcess::findBB(IplImage* imgSrc)
{
	CvRect aux;
	int xmin, xmax, ymin, ymax;
	xmin=xmax=ymin=ymax=0;

	findX(imgSrc, &xmin, &xmax);
	findY(imgSrc, &ymin, &ymax);

	aux=cvRect(xmin, ymin, xmax-xmin, ymax-ymin);

	printf("BB: %d,%d - %d,%d\n", aux.x, aux.y, aux.width, aux.height);

	return aux;
}

void PreProcess::findX(IplImage* imgSrc,int* min, int* max)
{
	int i;
	int minFound=0;
	CvMat data;
	CvScalar maxVal=cvRealScalar(imgSrc->height * 255);
	CvScalar val=cvRealScalar(0);
	//For each col sum, if sum < width*255 then we find the min
	//then continue to end to search the max, if sum< width*255 then is new max
	for (i=0; i< imgSrc->width; i++){
		cvGetCol(imgSrc, &data, i);
		val= cvSum(&data);
		if(val.val[0] < maxVal.val[0])
		{
			*max= i;
			if(!minFound)
			{
				*min= i;
				minFound= 1;
			}
		}
	}
}

void PreProcess::findY(IplImage* imgSrc,int* min, int* max)
{
	int i;
	int minFound=0;
	CvMat data;
	CvScalar maxVal=cvRealScalar(imgSrc->width * 255);
	CvScalar val=cvRealScalar(0);
	//For each col sum, if sum < width*255 then we find the min
	//then continue to end to search the max, if sum< width*255 then is new max
	for (i=0; i< imgSrc->height; i++){
		cvGetRow(imgSrc, &data, i);
		val= cvSum(&data);
		if(val.val[0] < maxVal.val[0]){
			*max=i;
			if(!minFound){
				*min= i;
				minFound= 1;
			}
		}
	}
}


IplImage* PreProcess::preprocessing(IplImage* imgSrc,int new_width, int new_height)
{
	IplImage* result;
	IplImage* scaledResult;

	CvMat data;
	CvMat dataA;
	CvRect bb;//bounding box
	CvRect bba;//bounding box maintain aspect ratio

	//Find bounding box
	bb=findBB(imgSrc);

	CvPoint pt1;
	CvPoint pt2;

	pt1.x = bb.x;
	pt1.y = bb.y;
	pt2.x = bb.width + bb.x;
	pt2.y = bb.height + bb.y;

	cvSetImageROI(imgSrc, cvRect(bb.x, bb.y, bb.width, bb.height));\

	IplImage* final = cvCreateImage(cvSize(bb.width,bb.height),imgSrc->depth,imgSrc->nChannels);
	cvCopy(imgSrc,final);
	return final;
}

void PreProcess::displaySobels(IplImage* img)
{
	//SOBEL ON THE X DERIVATIVE
	IplImage* df_dx = cvCreateImage(cvGetSize(img),IPL_DEPTH_16S, img->nChannels);
	cvSobel(img,df_dx,1,0,3);

	//SOBEL ON THE Y DERIVATIVE
	IplImage* df_dy = cvCreateImage(cvGetSize(img),IPL_DEPTH_16S, img->nChannels);
	cvSobel(img,df_dy,0,1,3);

	/* Convert signed to unsigned 8*/
	IplImage* dest_dx = cvCreateImage(cvGetSize(img),IPL_DEPTH_8U, img->nChannels);
	cvConvertScaleAbs( df_dx , dest_dx, 1, 0); 

	/* Convert signed to unsigned 8*/
	IplImage* dest_dy = cvCreateImage(cvGetSize(img),IPL_DEPTH_8U, img->nChannels);
	cvConvertScaleAbs( df_dy , dest_dy, 1, 0); 

	cvShowImage("x",dest_dx);
	cvShowImage("y",dest_dy);

	uchar* ptr;

	//NOW COMBINE BOTH TO FORM THE GRADIENT ANGLE MATRIX ANGLE = arctan(dx/dy)
	Mat gradientMat;
	for (int i = 0; i < dest_dx->height; i++) 
	{
		ptr = (uchar *) dest_dx->imageData + (i * dest_dx->widthStep);
		for (int j = 0; j < dest_dx->width; j++) 
		{
		//	printf("%d",ptr[j]); //<----- this will give you the pixel value at image[i][j]
		}
		//printf("\n");
	}
}