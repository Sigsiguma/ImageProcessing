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

void getLowPassDCT(Mat& filter, Size size, int r) {

	filter = Mat::zeros(size.height, size.width, CV_32F);

	for (int y = 0; y < size.height; ++y) {
		float *filterTmp = filter.ptr<float>(y);
		for (int x = 0; x < size.width; ++x) {
			if (x * x + y * y < r * r) {
				filterTmp[x] = 1;
			}
		}
	}
}

void LowPassFilter(const Mat& src, Mat& dest, int r) {
	dct(src, src);
	Mat lowPass;
	getLowPassDCT(lowPass, Size(src.cols, src.rows), r);
	imshow("LowPass", lowPass);
	dest = src.mul(lowPass);
	idct(dest, dest);
}

void gaussianFilter(const Mat& src, Mat& dest, double sigma) {
	dct(src, src);
	Mat gause;
	getGaussianMaskDCT(gause, Size(src.cols, src.rows), sigma);
	imshow("Gause", gause);
	dest = src.mul(gause);
	idct(dest, dest);
}

void gaussianNoiseDCT(const Mat& src, double sigma, int r) {
	Mat noise;
	addGaussianNoise(src, noise, 2.0);
	imshow("NOISE", noise);
	noise.convertTo(noise, CV_32FC1, 1.0 / 255);
	dct(noise, noise);
	Mat lowPass;
	getLowPassDCT(lowPass, Size(noise.cols, noise.rows), r);
	lowPass = noise.mul(lowPass);
	idct(lowPass, lowPass);
	noise.convertTo(noise, CV_8U);
	lowPass.convertTo(lowPass, CV_8U);
	cout << "PSNR: " << PSNR(noise, lowPass) << endl;
}


int main() {

	const string windowName = "Window";
	namedWindow(windowName, WINDOW_AUTOSIZE);

	Mat src = imread("./img/lenna.png", IMREAD_GRAYSCALE);
	src.convertTo(src, CV_32FC1, 1.0 / 255);
	Mat dest;
	src.copyTo(dest);

	gaussianFilter(src, dest, 3.0);

	gaussianNoiseDCT(src, 2.0, 100);

	while (1) {

		imshow(windowName, dest);

		if (waitKey(1) == 'q') {
			break;
		}
	}

	destroyAllWindows();

	return 0;
}
