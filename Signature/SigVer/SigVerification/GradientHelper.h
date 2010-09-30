///<summary>
///This class is a helper class header. It provides functions to calculate gradient values based on the Sobel Operator.
///It can process the entire image or an ROI.
///</summary>

#ifndef GradientHelper_H
#define GradientHelper_H
#include <cv.h>

namespace gsc
{

	class GradientHelper
	{
	public:
		GradientHelper::GradientHelper();
		void GradientHelper::getGradientMatrix(IplImage* src,cv::Mat* dest);
	};
}

using namespace std;

#endif /* GradientHelper_H */