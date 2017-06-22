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

class ParallelGaussian : public ParallelLoopBody {
public:
	ParallelGaussian(Mat& src, Mat& dest, int r, float sigma) : src_(src), dest_(dest), r_(r), sigma_(sigma) {}

	void operator()(const Range& range) const {
	}

private:
	Mat src_;
	Mat dest_;
	int r_;
	float sigma_;
};


void GaussianFilter_sugiura(const Mat &src, Mat &dest, int r, float sigma) {

	CV_Assert(src.size() == dest.size());
	CV_Assert(src.type() == CV_8UC3);
	CV_Assert(dest.type() == CV_8UC3);

	Mat srcExpandBoarder;
	//端のときに範囲外にならないようにフィルタ半径分全方向に広げる
	copyMakeBorder(src, srcExpandBoarder, r, r, r, r, cv::BORDER_REFLECT_101);

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

	for (int y = 0; y < src.rows; ++y) {
		Vec3b *destSrc = dest.ptr<Vec3b>(y);
		for (int x = 0; x < src.cols; ++x) {
			Mat partMat = srcExpandBoarder(Rect(x, y, 2 * r + 1, 2 * r + 1));
			const int kernelSize = 2 * r + 1;
			double result[3] = {0.0, 0.0, 0.0};
			for (int u = 0; u < kernelSize; ++u) {
				const Vec3b *imgSrc = partMat.ptr<Vec3b>(u);
				for (int v = 0; v < kernelSize; ++v) {
					result[0] += imgSrc[v][0] * kernel.at(u * kernelSize + v);
					result[1] += imgSrc[v][1] * kernel.at(u * kernelSize + v);
					result[2] += imgSrc[v][2] * kernel.at(u * kernelSize + v);
				}
			}

			destSrc[x][0] = (result[0] / sum);
			destSrc[x][1] = (result[1] / sum);
			destSrc[x][2] = (result[2] / sum);
		}
	}
}


void CheckTime(const Mat &src, Mat &dest, int r) {
	auto start = chrono::system_clock::now();
	GaussianFilter_sugiura(src, dest, r, r / 3.0f);
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

	int r = 5;
	int kernelSize = r * 2 + 1;

	CheckTime(src, dest, r);
	imshow("Gaussian", dest);

	Mat dest2;
	GaussianBlur(src, dest2, Size(kernelSize, kernelSize), r / 3.0);

	imshow("Gaussian2", dest2);

	cout << PSNR(dest, dest2) << endl;

	while (1) {
		if (waitKey(1) == 'q') {
			break;
		}
	}

	destroyAllWindows();

	return 0;
}
