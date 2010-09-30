#ifndef MeasureOfPenPressure_H
#define MeasureOfPenPressure_H
#include <cv.h>
#include <highgui.h>

using namespace cv;
namespace mopp
{
	
	class MeasureOfPenPressure
	{
		private:
		static double getThreshVal_Otsu_8u( const Mat& _src );

		public:
		
		//Simple Constructor
		MeasureOfPenPressure();

		float getEntropy(IplImage*);
		//To find Entropy, we need to create the histogram over L possible gray values and then take
		//E = Sum(-p*logp).

		int getGrayLevelThreshold(IplImage*);
		//To find GrayLevelThreshold, we need to hack the cvThreshold function to return us the k value.

		int getNumberOfBlackPixels(double,IplImage*);
		//To find NumberOfBlackPixels, we need find the number of pixels above k.
	};
}
using namespace std;

#endif /* MeasureOfPenPressure_H */