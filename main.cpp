#define _USE_MATH_DEFINES

#include <vector>
#include <numeric>
#include "include.hpp"
#include "sample_code.hpp"
#include "plot.hpp"

using namespace std;
using namespace cv;

template<typename T>
T clamp(const T &value, const T &low, const T &high) {
	return value < low ? low : (value > high ? high : value);
}

/*
Mat LowPassFilter(int row, int col) {

	int range = 10;
	double sigma = 10.0;
	Mat filter(row, col, CV_64FC1);

	for (int y = 0; y < row; ++y) {
		double *filterTmp = filter.ptr<double>(y);
		for (int x = 0; x < col; ++x) {
				filterTmp[x] =  exp(-2 * M_PI * M_PI * sigma * sigma * (x * x + y * y));
		}
	}

	return filter;
}
 */


int main() {

	const string windowName = "Window";
	namedWindow(windowName, WINDOW_AUTOSIZE);

	Mat src = imread("./img/lenna.png", IMREAD_GRAYSCALE);
	src.convertTo(src, CV_64FC1, 1.0 / 255);
	Mat dest;
	src.copyTo(dest);

	dct(src, src);
	imshow("DCT", src);

//	dest = src.mul(lowPass);

	idct(src, dest);


	while (1) {

		imshow(windowName, dest);

		if (waitKey(1) == 'q') {
			break;
		}
	}

	destroyAllWindows();

	return 0;
}
