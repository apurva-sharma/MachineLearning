#include "GradientHelper.h" 

using namespace std;
using namespace gsc;

GradientHelper::GradientHelper()
{
	printf("Constructor for GradientHelper");
}

void GradientHelper::getGradientMatrix(IplImage* imgSrc,cv::Mat* dest)
{
	//cv::Sobel(src,dest,1,1,1,,,,);

	int i,j;
	int minFound=0;
	CvScalar window[3][3];

	for (i=1; i < 2; i++)
	{
		for(j = 1; j < 2; j++)
		{
			//center pixel will be windowRows[1][j]
			for(int x = 0; x < 3; x++)
			{
				for(int y = 0; y < 3; y++)
				{
					window[x][y] = cvGet2D(imgSrc,i+(x-1),j+(y-1));
					//try printing the window here
					printf("%d \t",window[x][y]);
				}
				printf("\n");
			}
		}
	}

	//just print to see if you are getting the right values or not
	printf("\n");
	printf("\n");
	for(i=0;i<imgSrc->height;i++)
	{

		for(j=0;j<imgSrc->width;j++)
		{
			printf("%d \t",cvGet2D(imgSrc,i,j));
		}
		printf("\n");
	}
}
