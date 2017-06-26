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

/* �ȉ�ParallelLoopBody
void GaussianFilter(const Mat &src, Mat &dest, const float* kernel, const int r, const float sigma) {

	CV_Assert(src.size() == dest.size());

	Mat srcExpandBoarder;
	copyMakeBorder(src, srcExpandBoarder, r, r, r, r, BORDER_REFLECT101);

	for (int y = 0; y < src.rows; ++y) {
		Vec3f *destTmp = dest.ptr<Vec3f>(y);
		for (int x = 0; x < src.cols; ++x) {
			float resultB = 0.0f;
			float resultG = 0.0f;
			float resultR = 0.0f;

			for (int i = -r; i <= r; ++i) {
				const Vec3f *srcTmp = srcExpandBoarder.ptr<Vec3f>(y + r + i);
				resultB += srcTmp[r + x](0) * kernel[i + r];
				resultG += srcTmp[r + x](1) * kernel[i + r];
				resultR += srcTmp[r + x](2) * kernel[i + r];
			}

			destTmp[x](0) = resultB;
			destTmp[x](1) = resultG;
			destTmp[x](2) = resultR;
		}
	}

	Mat srcExpandBoarder2;
	copyMakeBorder(dest, srcExpandBoarder2, r, r, r, r, BORDER_REFLECT101);

	for (int y = 0; y < src.rows; ++y) {
		Vec3f *destTmp = dest.ptr<Vec3f>(y);
		const Vec3f *srcTmp = srcExpandBoarder2.ptr<Vec3f>(y + r);
		for (int x = 0; x < src.cols; ++x) {
			float resultB = 0.0f;
			float resultG = 0.0f;
			float resultR = 0.0f;

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
}

class ParallelGaussian : public ParallelLoopBody {
public:
	ParallelGaussian(const Mat &src, Mat &dest, float* kernel, int r, float sigma)
		: src_(src), dest_(dest), kernel_(kernel), r_(r), sigma_(sigma) {}

	void operator()(const Range &range) const {
		int row0 = range.start;
		int row1 = range.end;
		Mat srcStripe = src_.rowRange(row0, row1);
		Mat destStripe = dest_.rowRange(row0, row1);
		GaussianFilter(srcStripe, destStripe, kernel_, r_, sigma_);
	}

private:
	const Mat src_;
	const Mat dest_;
	const float* kernel_;
	const int r_;
	const float sigma_;
};

void GaussianFilter_sugiura(const Mat &src, Mat &dest, int r, float sigma) {

	float *kernel = new float[2 * r + 1];
	float sum = 0.0;
	{
		float sigma2 = sigma * sigma;

#pragma omp parallel for
		for (int x = -r; x <= r; ++x) {
			kernel[x + r] = exp(-(x * x) / (2.0f * sigma2));
			sum += kernel[x + r];
		}
	}

	sum = 1.0f / sum;
#pragma omp parallel for
	for (int x = -r; x <= r; ++x) {
		kernel[x + r] *= sum;
	}

	parallel_for_(Range(0, dest.rows), ParallelGaussian(src, dest, kernel, r, sigma), getNumThreads() - 1);

	delete[] kernel;
}
*/

void GaussianFilter_sugiura(const Mat& src, Mat& dest, const int r, const float sigma) {

	CV_Assert(src.size() == dest.size());
	dest = src.clone();

	float *kernel = new float[2 * r + 1]();
	float sum = 0.0f;
	{
		float sigma2 = sigma * sigma;
#pragma omp parallel for reduction(+:sum)
		for (int x = -r; x <= r; ++x) {
			kernel[x + r] = exp(-(x * x) / (2.0f * sigma2));
			sum += kernel[x + r];
		}
	}

	sum = 1.0f / sum;
#pragma omp parallel for
	for (int x = -r; x <= r; ++x) {
		kernel[x + r] *= sum;
	}

	Mat srcExpandBoarder;
	copyMakeBorder(src, srcExpandBoarder, r, r, r, r, BORDER_REFLECT101);

#pragma omp parallel for
	for (int y = 0; y < src.rows; ++y) {
		Vec3f *destTmp = dest.ptr<Vec3f>(y);
		for (int x = 0; x < src.cols; ++x) {
			float resultB = 0.0f;
			float resultG = 0.0f;
			float resultR = 0.0f;

			for (int i = -r; i <= r; ++i) {
				const Vec3f *srcTmp = srcExpandBoarder.ptr<Vec3f>(y + r + i);
				resultB += srcTmp[r + x](0) * kernel[i + r];
				resultG += srcTmp[r + x](1) * kernel[i + r];
				resultR += srcTmp[r + x](2) * kernel[i + r];
			}

			destTmp[x](0) = resultB;
			destTmp[x](1) = resultG;
			destTmp[x](2) = resultR;
		}
	}

	Mat srcExpandBoarder2;
	copyMakeBorder(dest, srcExpandBoarder2, r, r, r, r, BORDER_REFLECT101);

#pragma omp parallel for
	for (int y = 0; y < src.rows; ++y) {
		Vec3f *destTmp = dest.ptr<Vec3f>(y);
		const Vec3f *srcTmp = srcExpandBoarder2.ptr<Vec3f>(y + r);
		for (int x = 0; x < src.cols; ++x) {
			float resultB = 0.0f;
			float resultG = 0.0f;
			float resultR = 0.0f;

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

	delete[] kernel;
}

void CheckTime(const Mat src, Mat& dest, const int r, const float sigma, int count = 1) {
	auto start = chrono::system_clock::now();
	for (int i = 0; i < count; ++i) {
		GaussianFilter_sugiura(src, dest, r, sigma);
	}
	auto end = chrono::system_clock::now();
	auto time = end - start;
	auto msec = chrono::duration_cast<std::chrono::milliseconds>(time).count() / count;
	std::cout << "Time: " << msec << endl;
}

int main() {

	Mat src = imread("./img/Kodak/kodim03.png");
	imshow("SRC", src);
	src.convertTo(src, CV_32FC3, 1 / 255.0f);
	Mat dest;
	src.copyTo(dest);

	int r = 100;
	int kernelSize = r * 2 + 1;

	CheckTime(src, dest, r, r / 3.0f, 1);

	Mat dest2;
	GaussianBlur(src, dest2, Size(kernelSize, kernelSize), r / 3.0f);

	dest.convertTo(dest, CV_8UC3, 255);
	dest2.convertTo(dest2, CV_8UC3, 255);
	imshow("MyGaussian", dest);
	imshow("Gaussian", dest2);

	cout << "PSNR:" << PSNR(dest, dest2) << endl;

	while (1) {
		if (waitKey(1) == 'q') {
			break;
		}
	}

	destroyAllWindows();

	return 0;
}
