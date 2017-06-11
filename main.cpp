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
	copyMakeBorder(src, srcExpandBoarder, r, r, r, r, cv::BORDER_REFLECT_101);

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

void lowPassFilter(const Mat src, Mat &dest, int r) {
	Mat dctSrc;
	dct(src, dctSrc);
	Mat lowPass;
	getLowPassDCT(lowPass, Size(src.cols, src.rows), r);
	dest = dctSrc.mul(lowPass);
	idct(dest, dest);
	imshow("LowPassFilter", dest);
}

void gaussianFilter(const Mat src, Mat &dest, double sigma) {
	Mat dctSrc;
	dct(src, dctSrc);
	Mat gause;
	getGaussianMaskDCT(gause, Size(src.cols, src.rows), sigma);
	dest = dctSrc.mul(gause);
	idct(dest, dest);
	imshow("GaussianFilter", dest);
//	dest.convertTo(dest, CV_8U, 255);
//	imwrite("Gaussian.png", dest);
}

void gaussianNoiseDCT(const Mat src, double sigma, int r) {
	Mat srcU;
	src.convertTo(srcU, CV_8U, 255);
	Mat noise;
	addGaussianNoise(srcU, noise, sigma);
	noise.convertTo(noise, CV_32F, 1.0 / 255);
	Mat result;
	lowPassFilter(noise, result, r);
	result.convertTo(result, CV_8U, 255);
	imshow("GaussianNoiseDCT", result);
	cout << "PSNR: " << PSNR(srcU, result) << endl;
}


void spikeNoiseDCT(const Mat src, double noise_rate, int r) {
	Mat srcU;
	src.convertTo(srcU, CV_8U, 255);
	Mat noise;
	addSpikeNoise(srcU, noise, noise_rate);
	noise.convertTo(noise, CV_32F, 1.0 / 255);
	Mat result;
	lowPassFilter(noise, result, r);
	result.convertTo(result, CV_8U, 255);
	imshow("SpikeNoiseDCT", result);
	cout << "PSNR:" << PSNR(srcU, result) << endl;
}

void gaussianNoiseMedian(const Mat src, double sigma, int kernel) {
	Mat srcU;
	src.convertTo(srcU, CV_8U, 255);
	Mat noise;
	addGaussianNoise(srcU, noise, sigma);
	Mat result;
	medianBlur(noise, result, kernel);
	imshow("GaussianNoiseMedian", result);
	cout << "PSNR:" << PSNR(srcU, result) << endl;
}

void spikeNoiseMedian(const Mat src, double noise_rate, int kernel) {
	Mat srcU;
	src.convertTo(srcU, CV_8U, 255);
	Mat noise;
	addSpikeNoise(srcU, noise, noise_rate);
	Mat result;
	medianBlur(noise, result, kernel);
	imshow("SpikeNoiseMedian", result);
	cout << "PSNR:" << PSNR(srcU, result) << endl;
}

void gaussianNoiseBilateral(const Mat src, double sigma, int r) {
	Mat srcU;
	src.convertTo(srcU, CV_8U, 255);
	Mat noise;
	addGaussianNoise(srcU, noise, sigma);
	Mat result;
	noise.copyTo(result);
	bilateralFilter(noise, result, r, 15.0);
	imshow("GaussianNoiseBilateral", result);
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
	imshow("SpikeNoiseBilateral", result);
	cout << "PSNR:" << PSNR(srcU, result) << endl;
	imwrite("SpikeNoiseBilateral.png", result);
}

enum class MethodType {
	LowPassFilter,
	GaussianFilter,
	GaussianNoiseDCT,
	SpikeNoiseDCT,
	GaussianNoiseMedian,
	SpikeNoiseMedian,
	GaussianNoiseBilateral,
	SpikeNoiseBilateral
};


int main() {

	const string windowName = "Window";
	namedWindow(windowName, WINDOW_AUTOSIZE);

	Mat src = imread("./img/Kodak/kodim23.png", IMREAD_GRAYSCALE);
	src.convertTo(src, CV_32F, 1.0 / 255);
	Mat dest;
	src.copyTo(dest);

	int type;
	cout << "0: LowPassFilter, 1: GaussianFilter, 2: GaussianNoiseDCt, 3: SpikeNoiseDCT, 4: GaussianNoiseMedian"
	     << endl;
	cout << "5: SpikeNoiseMedian, 6: GaussianNoiseBilateral, 7: SpikeNoiseBilateral" << endl;
	cin >> type;

	switch (static_cast<MethodType>(type)) {
		case MethodType::LowPassFilter:
			lowPassFilter(src, dest, 100);
			break;
		case MethodType::GaussianFilter:
			gaussianFilter(src, dest, 3.0);
			break;
		case MethodType::GaussianNoiseDCT:
			gaussianNoiseDCT(src, 15.0, 100);
			break;
		case MethodType::SpikeNoiseDCT:
			spikeNoiseDCT(src, 20.0, 100);
			break;
		case MethodType::GaussianNoiseMedian:
			gaussianNoiseMedian(src, 15.0, 13);
			break;
		case MethodType::SpikeNoiseMedian:
			spikeNoiseMedian(src, 20.0, 13);
			break;
		case MethodType::GaussianNoiseBilateral:
			gaussianNoiseBilateral(src, 15.0, 6);
			break;
		case MethodType::SpikeNoiseBilateral:
			spikeNoiseBilateral(src, 20.0, 6);
			break;
	}

	while (1) {

		if (waitKey(1) == 'q') {
			break;
		}
	}

	destroyAllWindows();

	return 0;
}
