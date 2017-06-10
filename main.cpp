#define _USE_MATH_DEFINES

#include <vector>
#include <numeric>
#include "include.hpp"
#include "sample_code.hpp"
#include "plot.hpp"

using namespace std;
using namespace cv;

double pixelValueWeight(const Mat &src, const int m, const int n, const int x, const int y, const double space_sigma,
                        const double color_sigma) {

	CV_Assert(src.type() == CV_8UC1);

	const uchar fxy = src.ptr<uchar>(y)[x];
	const uchar fxmyn = src.ptr<uchar>(y + n)[x + m];

	double weight = exp(-(m * m + n * n) / (2 * space_sigma * space_sigma)) *
	                exp((-(fxy - fxmyn) * (fxy - fxmyn)) / 2 * color_sigma * color_sigma);


	return weight;
}

void bilateralFilter(const Mat &src, Mat &dest, int r, double color_sigma) {

	CV_Assert(src.size() == dest.size());
	CV_Assert(src.type() == CV_8UC1);
	CV_Assert(dest.type() == CV_8UC1);

	double sigma = r / 3.0;

	const int kernelSize = 2 * r + 1;

	Mat srcExpandBoarder;
	copyMakeBorder(src, srcExpandBoarder, r + 1, r + 1, r + 1, r + 1, cv::BORDER_REFLECT_101);

	for (int y = 0; y < src.rows; ++y) {
		uchar *imgDest = dest.ptr<uchar>(y);
		for (int x = 0; x < src.cols; ++x) {

			double denominator = 0.0;
			double numerator = 0.0;
			for (int n = 0; n < kernelSize; ++n) {
				const uchar *imgSrc = srcExpandBoarder.ptr<uchar>(y + n);
				for (int m = 0; m < kernelSize; ++m) {
					denominator += pixelValueWeight(srcExpandBoarder, m, n, x, y, sigma, color_sigma);
					numerator += imgSrc[x + m] * pixelValueWeight(srcExpandBoarder, m, n, x, y, sigma, color_sigma);
				}
			}
			imgDest[x] = numerator / denominator;
		}
	}
}

void getLowPassDCT(Mat &filter, Size size, int r) {

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

void LowPassFilter(const Mat src, Mat &dest, int r) {
	Mat dctSrc;
	dct(src, dctSrc);
	Mat lowPass;
	getLowPassDCT(lowPass, Size(src.cols, src.rows), r);
	dest = dctSrc.mul(lowPass);
	idct(dest, dest);
}

void gaussianFilter(const Mat src, Mat &dest, double sigma) {
	Mat dctSrc;
	dct(src, dctSrc);
	Mat gause;
	getGaussianMaskDCT(gause, Size(src.cols, src.rows), sigma);
	dest = dctSrc.mul(gause);
	idct(dest, dest);
}

void gaussianNoiseDCT(const Mat src, double sigma, int r) {
	Mat noise;
	addGaussianNoise(src, noise, sigma);
	noise.convertTo(noise, CV_32FC1);
	Mat result;
	LowPassFilter(noise, result, r);
	Mat srcU;
	src.convertTo(srcU, CV_8U, 255);
	result.convertTo(result, CV_8U, 255);
	cout << "PSNR: " << PSNR(srcU, result) << endl;
}

void spikeNoiseDCT(const Mat src, double noise_rate, int r) {
	Mat srcU;
	src.convertTo(srcU, CV_8U, 255);
	Mat noise;
	addSpikeNoise(srcU, noise, noise_rate);
	noise.convertTo(noise, CV_32FC1, 1.0 / 255);
	Mat result;
	LowPassFilter(noise, result, r);
	result.convertTo(result, CV_8U, 255);
	cout << "PSNR:" << PSNR(srcU, result) << endl;
}

void gaussianNoiseMedian(const Mat src, double sigma, int kernel) {
	Mat noise;
	addGaussianNoise(src, noise, sigma);
	noise.convertTo(noise, CV_8U, 255);
	Mat result;
	medianBlur(noise, result, kernel);
	Mat srcU;
	src.convertTo(srcU, CV_8U, 255);
	cout << "PSNR:" << PSNR(srcU, result) << endl;
}

void spikeNoiseMedian(const Mat src, double noise_rate, int kernel) {
	Mat srcU;
	src.convertTo(srcU, CV_8U, 255);
	Mat noise;
	addSpikeNoise(srcU, noise, noise_rate);
	Mat result;
	medianBlur(noise, result, kernel);
	cout << "PSNR:" << PSNR(srcU, result) << endl;
}

void gaussianNoiseBilateral(const Mat src, double sigma, int r) {
	Mat noise;
	addGaussianNoise(src, noise, sigma);
	noise.convertTo(noise, CV_8U, 255);
	Mat result;
	noise.copyTo(result);
	bilateralFilter(noise, result, r, 0.01);
	Mat srcU;
	src.convertTo(srcU, CV_8U, 255);
	cout << "PSNR:" << PSNR(srcU, result) << endl;
}

void spikeNoiseBilateral(const Mat src, double noise_rate, int r) {
	Mat srcU;
	src.convertTo(srcU, CV_8U, 255);
	Mat noise;
	addSpikeNoise(srcU, noise, noise_rate);
	Mat result;
	noise.copyTo(result);
	bilateralFilter(noise, result, r, 0.01);
	cout << "PSNR:" << PSNR(srcU, result) << endl;
}


int main() {

	const string windowName = "Window";
	namedWindow(windowName, WINDOW_AUTOSIZE);

	Mat src = imread("./img/Kodak/kodim23.png", IMREAD_GRAYSCALE);
	src.convertTo(src, CV_32FC1, 1.0 / 255);
	Mat dest;
	src.copyTo(dest);

	gaussianFilter(src, dest, 3.0);

//	gaussianNoiseDCT(src, 1.0, 200);

//	spikeNoiseDCT(src, 20.0, 100);

//	gaussianNoiseMedian(src, 1.0, 13);

//	spikeNoiseMedian(src, 20.0, 13);

//	gaussianNoiseBilateral(src, 1.0, 6);

	spikeNoiseBilateral(src, 20.0, 6);

	while (1) {

		imshow(windowName, dest);

		if (waitKey(1) == 'q') {
			break;
		}
	}

	destroyAllWindows();

	return 0;
}
