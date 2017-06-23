#define _USE_MATH_DEFINES

#include <vector>
#include <numeric>
#include <algorithm>
#include <chrono>
#include "include.hpp"
#include "sample_code.hpp"
#include "plot.hpp"

using namespace std;
using namespace cv;

void GaussianFilter(const Mat &src, Mat &dest, const int r, const float sigma) {

	CV_Assert(src.size() == dest.size());

	Mat srcExpandBoarder;
	//端のときに範囲外にならないようにフィルタ半径分全方向に広げる
	copyMakeBorder(src, srcExpandBoarder, r, r, r, r, cv::BORDER_REFLECT_101);

	double data[2 * r + 1];
	{
		float sigma2 = sigma * sigma;

		for (int x = -r; x <= r; ++x) {
			data[x + r] = sqrt(exp(-(x * x + x * x) / (2.0 * sigma2)));
		}
	}

	vector<double> kernel;
	{
		float sigma2 = sigma * sigma;

		for (int y = -r; y <= r; ++y) {
			for (int x = -r; x <= r; ++x) {
				kernel.emplace_back(exp(-(x * x + y * y) / (2.0 * sigma2)));
			}
		}
	}
	double sum = std::accumulate(kernel.begin(), kernel.end(), 0.0);

	Mat matA(1, 2 * r + 1, CV_64FC1, data);
	Mat matB(2 * r + 1, 1, CV_64FC1, data);


	for (int y = 0; y < src.rows; ++y) {
		Vec3d *destSrc = dest.ptr<Vec3d>(y);
		for (int x = 0; x < src.cols; ++x) {
			Mat partMat = srcExpandBoarder(Rect(x, y, 2 * r + 1, 2 * r + 1));
			vector<Mat> colors;
			split(partMat, colors);

			vector<Mat> results;
			results.emplace_back(matA * colors[0] * matB);
			results.emplace_back(matA * colors[1] * matB);
			results.emplace_back(matA * colors[2] * matB);


			Mat result;
			merge(results, result);

			destSrc[x] = result.ptr<Vec3d>(0)[0] / sum;
		}
	}
}


class ParallelGaussian : public ParallelLoopBody {
public:
	ParallelGaussian(const Mat &src, Mat &dest, int r, float sigma) : src_(src), dest_(dest), r_(r), sigma_(sigma) {}

	void operator()(const Range &range) const {
		int row0 = range.start;
		int row1 = range.end;
		Mat srcStripe = src_.rowRange(row0, row1);
		Mat destStripe = dest_.rowRange(row0, row1);
		GaussianFilter(srcStripe, destStripe, r_, sigma_);
	}

private:
	const Mat src_;
	const Mat dest_;
	const int r_;
	const float sigma_;
};

void GaussianFilter_sugiura(const Mat &src, Mat &dest, int r, float sigma) {
	parallel_for_(Range(0, dest.rows), ParallelGaussian(src, dest, r, r / 3.0f));
}

void CheckTime(Mat &src, Mat &dest, int r) {
	auto start = chrono::system_clock::now();
//	GaussianFilter(src, dest, r, r / 3.0f);
	GaussianFilter_sugiura(src, dest, r, r / 3.0f);
	auto end = chrono::system_clock::now();
	auto time = end - start;
	auto msec = chrono::duration_cast<std::chrono::milliseconds>(time).count();
	std::cout << "Time: " << msec << endl;
}


int main() {

	Mat src = imread("./img/Kodak/kodim03.png");
	Mat srcD;
	src.convertTo(srcD, CV_64FC3, 1 / 255.0);
	Mat dest;
	srcD.copyTo(dest);
	imshow("SRC", src);


	int r = 10;
	int kernelSize = r * 2 + 1;

	CheckTime(srcD, dest, r);

	Mat dest2;
	GaussianBlur(src, dest2, Size(kernelSize, kernelSize), r / 3.0);
	imshow("Gaussian2", dest2);

	dest.convertTo(dest, CV_8UC3, 255);
	imshow("Gaussian", dest);
	cout << "PSNR:" << PSNR(dest, dest2) << endl;

	while (1) {
		if (waitKey(1) == 'q') {
			break;
		}
	}

	destroyAllWindows();

	return 0;
}
