#ifndef PreProcessing_H
#define PreProcessing_H
#include <cv.h>

using namespace cv;
namespace prep
{

	class PreProcess
	{

	public:

		//Simple Constructor
		PreProcess();

		CvRect findBB(IplImage* imgSrc);
		void findX(IplImage* imgSrc,int* min, int* max);
		void findY(IplImage* imgSrc,int* min, int* max);
		IplImage* preprocessing(IplImage* imgSrc,int new_width, int new_height);
		void displaySobels(IplImage* imgSrc);

	};
}
using namespace std;

#endif /* PreProcessing_H */