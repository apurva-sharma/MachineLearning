#include "MeasureOfPenPressure.h" 
#include <cv.h>
#include <highgui.h>
#include <math.h>

using namespace std;
using namespace mopp;

MeasureOfPenPressure::MeasureOfPenPressure()
{
}

float MeasureOfPenPressure::getEntropy(IplImage* imgSrc)
{
	//To find Entropy, we need to create the histogram over L possible gray values and then take
	//E = Sum(-p*logp).

	CvHistogram* hist;

	//size of the histogram -1D histogram
	int bins = 256;
	int hsize[] = {bins};
	float max_value = 0, min_value = 0;
	//value and normalized value
	float value;
	int normalized;

	//ranges - grayscale 0 to 256
	float xranges[] = { 0, 256 };
	float* ranges[] = { xranges };

	//create an 8 bit single channel image to hold a
	//grayscale version of the original picture
	IplImage* gray = cvCreateImage( cvGetSize(imgSrc), 8, 1 );
	cvCvtColor( imgSrc, gray, CV_BGR2GRAY );

	//planes to obtain the histogram, in this case just one
	IplImage* planes[] = { gray };

	//get the histogram and some info about it
	hist = cvCreateHist( 1, hsize, CV_HIST_ARRAY, ranges,1);
	cvCalcHist( planes, hist, 0, NULL);
	cvGetMinMaxHistValue( hist, &min_value, &max_value);
//	printf("min: %f, max: %f\n", min_value, max_value);
	

	//create an 8 bits single channel image to hold the histogram
	//paint it white
	IplImage* imgHistogram = cvCreateImage(cvSize(bins, 50),8,1);
	cvRectangle(imgHistogram, cvPoint(0,0), cvPoint(256,50), CV_RGB(255,255,255),-1);

	//draw the histogram :P
	long double p;
	float E = 0;
	long total = imgSrc->width * imgSrc->height;
	
	for(int i=0; i < bins; i++)
	{	
		p = 0.0;
		value = cvQueryHistValue_1D( hist, i);
		//normalized = cvRound(value*50/max_value);
		p = (double) value/total;
		float temp = (value == 0)? -1*(1/total)*log((double)1/total) : -1*p*log(p);
		E += temp;
		//cvLine(imgHistogram,cvPoint(i,50), cvPoint(i,50-normalized), CV_RGB(0,0,0));
	}
//	printf("the entropy is: %f\n",E);

	return E;
}

int MeasureOfPenPressure::getGrayLevelThreshold(IplImage* imgSrc)
{
	//To find GrayLevelThreshold, we need to hack the cvThreshold function to return us the k value.
	double k = getThreshVal_Otsu_8u(imgSrc);
//	printf("Threshold is : %f\n",k);
	return k;
}

int MeasureOfPenPressure::getNumberOfBlackPixels(double threshold, IplImage* imgSrc)
{
	//To find NumberOfBlackPixels, we need find the number of pixels above k.
	
	//this pointer will store the current value of the pixel from the image.
	uchar* ptr;
	//total number of pixels with value above threshold
	int nKplusCounter = 0;

	for (int i = 0; i < imgSrc->height; i++) 
	{
		ptr = (uchar *) imgSrc->imageData + (i * imgSrc->widthStep);
		for (int j = 0; j < imgSrc->width; j++) 
		{
			//ptr[j] <--- this will give you the pixel value at image[i][j]
			if(ptr[j] < threshold)
				nKplusCounter++;
		}
	}

	//cvShowImage("hello",imgSrc);
//	printf("total no of pixels is : %d\n",imgSrc->height * imgSrc->width);
//	printf("no of black pixels is : %d\n",nKplusCounter);
	return nKplusCounter;
}

double MeasureOfPenPressure::getThreshVal_Otsu_8u( const Mat& _src )
{
    Size size = _src.size();
    if( _src.isContinuous() )
    {
        size.width *= size.height;
        size.height = 1;
    }
    const int N = 256;
    int i, j, h[N] = {0};
    for( i = 0; i < size.height; i++ )
    {
        const uchar* src = _src.data + _src.step*i;
        for( j = 0; j <= size.width - 4; j += 4 )
        {
            int v0 = src[j], v1 = src[j+1];
            h[v0]++; h[v1]++;
            v0 = src[j+2]; v1 = src[j+3];
            h[v0]++; h[v1]++;
        }
        for( ; j < size.width; j++ )
            h[src[j]]++;
    }

    double mu = 0, scale = 1./(size.width*size.height);
    for( i = 0; i < N; i++ )
        mu += i*h[i];
    
    mu *= scale;
    double mu1 = 0, q1 = 0;
    double max_sigma = 0, max_val = 0;

    for( i = 0; i < N; i++ )
    {
        double p_i, q2, mu2, sigma;

        p_i = h[i]*scale;
        mu1 *= q1;
        q1 += p_i;
        q2 = 1. - q1;

        if( min(q1,q2) < FLT_EPSILON || max(q1,q2) > 1. - FLT_EPSILON )
            continue;

        mu1 = (mu1 + i*p_i)/q1;
        mu2 = (mu - q1*mu1)/q2;
        sigma = q1*q2*(mu1 - mu2)*(mu1 - mu2);
        if( sigma > max_sigma )
        {
            max_sigma = sigma;
            max_val = i;
        }
    }

    return max_val;
}
