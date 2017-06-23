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
	copyMakeBorder(src, srcExpandBoarder, r, r, r, r, BORDER_REFLECT101);

	double *kernel = new double[2 * r + 1];
	double sum = 0.0;
	{
		float sigma2 = sigma * sigma;

		for (int x = -r; x <= r; ++x) {
			kernel[x + r] = exp(-(x * x) / (2.0 * sigma2));
			sum += kernel[x + r];
		}
	}

	sum = 1.0 / sum;


	for (int x = -r; x <= r; ++x) {
		kernel[x + r] *= sum;
	}

	for (int y = 0; y < src.rows; ++y) {
		Vec3b *destTmp = dest.ptr<Vec3b>(y);
		const Vec3b *srcTmp = srcExpandBoarder.ptr<Vec3b>(y + r);
		for (int x = 0; x < src.cols; ++x) {
			double resultB = 0.0;
			double resultG = 0.0;
			double resultR = 0.0;

			for (int i = -r; i <= r; ++i) {
				resultB += srcTmp[r + x + i](0) * kernel[i + r];
				resultG += srcTmp[r + x + i](1) * kernel[i + r];
				resultR += srcTmp[r + x + i](2) * kernel[i + r];
			}

			destTmp[x](0) = resultB;
			destTmp[x](1) = resultG;
			destTmp[x](2) = resultR;
		}
	}

	Mat srcExpandBoarder2;
	copyMakeBorder(dest, srcExpandBoarder2, r, r, r, r, BORDER_REFLECT101);

	//ここから下のどこかがおかしい
	for (int y = 0; y < src.rows; ++y) {
		Vec3b *destTmp = dest.ptr<Vec3b>(y);
		for (int x = 0; x < src.cols; ++x) {
			double resultB = 0.0;
			double resultG = 0.0;
			double resultR = 0.0;

			for (int i = -r; i <= r; ++i) {
				const Vec3b *srcTmp = srcExpandBoarder2.ptr<Vec3b>(y + r + i);
				resultB += srcTmp[r + x](0) * kernel[i + r];
				resultG += srcTmp[r + x](1) * kernel[i + r];
				resultR += srcTmp[r + x](2) * kernel[i + r];
			}

			destTmp[x](0) = resultB;
			destTmp[x](1) = resultG;
			destTmp[x](2) = resultR;
		}
	}

	delete[] kernel;
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
	parallel_for_(Range(0, dest.rows), ParallelGaussian(src, dest, r, sigma));
}

void CheckTime(Mat &src, Mat &dest, int r, float sigma) {
	auto start = chrono::system_clock::now();
//	GaussianFilter(src, dest, r, r / 3.0f);
	GaussianFilter_sugiura(src, dest, r, sigma);
	auto end = chrono::system_clock::now();
	auto time = end - start;
	auto msec = chrono::duration_cast<std::chrono::milliseconds>(time).count();
	std::cout << "Time: " << msec << endl;
}


int main() {

	Mat src = imread("./img/Kodak/kodim03.png");
	Mat dest;
	src.copyTo(dest);
	imshow("SRC", src);

	int r = 40;
	int kernelSize = r * 2 + 1;

	CheckTime(src, dest, r, r / 3.0f);
	imshow("Gaussian1", dest);

	Mat dest2;
	GaussianBlur(src, dest2, Size(1, kernelSize), r / 3.0f);
	imshow("Gaussian2", dest2);

	cout << "PSNR:" << PSNR(dest, dest2) << endl;

	while (1) {
		if (waitKey(1) == 'q') {
			break;
		}
	}

	destroyAllWindows();

	return 0;
}
