#define _USE_MATH_DEFINES

#include <vector>
#include "include.hpp"
#include "sample_code.hpp"
#include "plot.hpp"

using namespace std;
using namespace cv;

template<typename T>
T clamp(const T &value, const T &low, const T &high) {
	return value < low ? low : (value > high ? high : value);
}


uchar filter(const Mat &src, const vector<double> &kernel, int r) {

	double result = 0;

	const int kernelSize = 2 * r + 1;

	for (int y = 0; y < kernelSize; ++y) {
		const uchar *imgSrc = src.ptr<uchar>(y);
		for (int x = 0; x < kernelSize; ++x) {
			result += imgSrc[x] * kernel.at(y * kernelSize + x);
		}
	}

	return clamp(static_cast<int>(result), 0, 255);
}

vector<double> createGaussianKernel(int r) {

	vector<double> data;

	double sigma = r / 3.0;
	double sigma2 = sigma * sigma;

	for (int y = -r; y <= r; ++y) {
		for (int x = -r; x <= r; ++x) {
			data.emplace_back((1.0 / (2.0 * M_PI * sigma2)) *
			                  exp(-(x * x + y * y) / (2.0 * sigma2)));
		}
	}

	return data;
}


void gaussianFilter(const Mat &src, Mat &dest, int r) {

	CV_Assert(src.size() == dest.size());
	CV_Assert(src.type() == CV_8UC1);
	CV_Assert(dest.type() == CV_8UC1);

	Mat srcExpandBoarder;
	//端のときに範囲外にならないようにフィルタ半径分全方向に広げる
	copyMakeBorder(src, srcExpandBoarder, r + 1, r + 1, r + 1, r + 1, cv::BORDER_REFLECT_101);

	double sigma = r / 3.0;

	vector<double> data = createGaussianKernel(r);

	for (int y = 0; y < src.rows; ++y) {
		uchar *imgDest = dest.ptr<uchar>(y);
		for (int x = 0; x < src.cols; ++x) {
			Mat partMat = srcExpandBoarder(Rect(x, y, 2 * r + 1, 2 * r + 1));
			imgDest[x] = filter(partMat, data, r);
		}
	}
}

void laplacianFilter(const Mat &src, Mat &dest) {

	CV_Assert(src.size() == dest.size());
	CV_Assert(src.type() == CV_8UC1);
	CV_Assert(dest.type() == CV_8UC1);

	int r = 1;

	Mat srcExpandBoarder;
	//端のときに範囲外にならないようにフィルタ半径分全方向に広げる
	copyMakeBorder(src, srcExpandBoarder, r + 1, r + 1, r + 1, r + 1, cv::BORDER_REFLECT_101);

	vector<double> data = {0, 1, 0, 1, -4, 1, 0, 1, 0};


	for (int y = 0; y < src.rows; ++y) {
		uchar *imgDest = dest.ptr<uchar>(y);
		for (int x = 0; x < src.cols; ++x) {
			Mat partMat = srcExpandBoarder(Rect(x, y, 2 * r + 1, 2 * r + 1));
			imgDest[x] = filter(partMat, data, r);
		}
	}

}

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
	//端のときに範囲外にならないようにフィルタ半径分全方向に広げる
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

void UnsharpMask(const Mat &src, Mat &dest, int r, int k) {

	CV_Assert(src.size() == dest.size());

	gaussianFilter(src, dest, r);

	Mat diff = dest - src;

	dest = src + k * diff;
}

enum class FilterType {
	Gaussian,
	Laplacian,
	Bilateral,
	Unsharp
};

int main() {

	const string windowName = "Window";
	const string trackBarName1 = "FilterRadius";
	const string trackBarName2 = "SharpeningCoefficient";
	namedWindow(windowName, WINDOW_AUTOSIZE);


	Mat img = imread("./img/lenna.png", CV_8UC1);
	Mat dest;
	img.copyTo(dest);

	int type;
	cout << "Input - Gaussian: 0, Laplacian: 1, Bilateral: 2, Unsharp: 3" << endl;
	cin >> type;

	int r = 2;
	int k = 1;

	switch (static_cast<FilterType>(type)) {
		case FilterType::Gaussian:
			gaussianFilter(img, dest, 6);
			break;
		case FilterType::Laplacian:
			laplacianFilter(img, dest);
			break;
		case FilterType::Bilateral:
			bilateralFilter(img, dest, 10, 0.1);
			break;
		case FilterType::Unsharp:
			cv::createTrackbar(trackBarName1, windowName, &r, 10);
			cv::createTrackbar(trackBarName2, windowName, &k, 9);
			UnsharpMask(img, dest, r, k);
			break;
	}

	while (1) {

		if (static_cast<FilterType>(type) == FilterType::Unsharp) {
			r = cv::getTrackbarPos(trackBarName1, windowName);
			k = cv::getTrackbarPos(trackBarName2, windowName);
			UnsharpMask(img, dest, r, k);
		}

		imshow(windowName, dest);

		if (waitKey(1) == 'q') {
			break;
		}
	}

	destroyAllWindows();

	return 0;
}
