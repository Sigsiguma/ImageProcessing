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

void nearestNeighbor(const Mat &src, Mat &dest, double xscale, double yscale) {

	dest = Mat(src.rows * yscale, src.cols * xscale, CV_8UC3);

	for (int y = 0; y < dest.rows; y++) {
		double srcY = static_cast<double>(y) / yscale + 0.5;
		const Vec3b *srcTmp = src.ptr<Vec3b>(static_cast<int>(srcY));
		Vec3b *destTmp = dest.ptr<Vec3b>(y);
		for (int x = 0; x < dest.cols; x++) {
			double srcX = static_cast<double>(x) / xscale + 0.5;

			//int型のキャストで切り捨て
			destTmp[x][0] = srcTmp[static_cast<int>(srcX)][0];
			destTmp[x][1] = srcTmp[static_cast<int>(srcX)][1];
			destTmp[x][2] = srcTmp[static_cast<int>(srcX)][2];
		}
	}
}

void bilinear(const Mat &src, Mat &dest, double xscale, double yscale) {

	dest = Mat(src.rows * yscale, src.cols * xscale, CV_8UC3);

	for (int y = 0; y < dest.rows; y++) {
		double srcY = static_cast<double>(y) / yscale;
		int floorY = static_cast<int>(srcY);
		const Vec3b *srcTmp1 = src.ptr<Vec3b>(floorY);
		const Vec3b *srcTmp2 = src.ptr<Vec3b>(floorY + 1);
		Vec3b *destTmp = dest.ptr<Vec3b>(y);
		for (int x = 0; x < dest.cols; x++) {
			double srcX = static_cast<double>(x) / xscale;
			int floorX = static_cast<int>(srcX);

			for (int color = 0; color < 3; ++color) {
				destTmp[x][color] = (floorX + 1 - srcX) * (floorY + 1 - srcY) * srcTmp1[floorX][color] +
				                (floorX + 1 - srcX) * (srcY - floorY) * srcTmp2[floorX][color] +
				                (srcX - floorX) * (floorY + 1 - srcY) * srcTmp1[floorX + 1][color] +
				                (srcX - floorX) * (srcY - floorY) * srcTmp2[floorX + 1][color];
			}
		}
	}

}

void rescale(const Mat &src, Mat &dest, double xscale, double yscale, int interpolation) {
	if (interpolation == CV_INTER_NN) {
		nearestNeighbor(src, dest, xscale, yscale);
	} else if (interpolation == CV_INTER_LINEAR) {
		bilinear(src, dest, xscale, yscale);
	}
}

void euclideanTransform(const Mat &src, Mat &dest, double theta, double tx, double ty) {

}

int main() {

	const string windowName = "Window";
	namedWindow(windowName, WINDOW_AUTOSIZE);


	Mat src = imread("./img/lenna.png");
	Mat dest;

	rescale(src, dest, 0.5, 0.5, CV_INTER_LINEAR);
	imshow(windowName, dest);
	resize(src, dest, Size(), 0.5, 0.5, CV_INTER_LINEAR);
	imshow("Method", dest);

	while (1) {

		if (waitKey(1) == 'q') {
			break;
		}
	}

	destroyAllWindows();

	return 0;
}
