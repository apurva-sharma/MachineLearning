#ifndef ChainCodeExtractor_H
#define ChainCodeExtractor_H

#include <cv.h>

namespace chainCode
{
	class ChainCodeExtractor
	{
	private:
		
		IplImage* img;
		int count, total, retrievalMethod;
		
	public:
		//simple constructor
		ChainCodeExtractor();
	
		void initChainCodeReader(IplImage* imgSrc,  int retrievalMethod);
		int getTotalContours(int retrievalMethod);
		int** getNextChainCodes(bool supressDebug);
	};
}
using namespace std;

#endif /* ChainCodeExtractor_H */