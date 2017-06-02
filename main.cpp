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
	Mat srcCpy;
	copyMakeBorder(src, srcCpy, 1, 1, 1, 1, BORDER_REFLECT_101);

	for (int y = 0; y < dest.rows; y++) {
		double srcY = static_cast<double>(y) / yscale + 0.5;
		const Vec3b *srcTmp = srcCpy.ptr<Vec3b>(static_cast<int>(srcY));
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
	Mat srcCpy;
	copyMakeBorder(src, srcCpy, 1, 1, 1, 1, BORDER_REFLECT_101);

	for (int y = 0; y < dest.rows; y++) {
		double srcY = static_cast<double>(y) / yscale;
		int floorY = static_cast<int>(srcY);
		const Vec3b *srcTmp1 = srcCpy.ptr<Vec3b>(floorY);
		const Vec3b *srcTmp2 = srcCpy.ptr<Vec3b>(floorY + 1);
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

	CV_Assert(xscale != 0.0 && yscale != 0.0);

	if (interpolation == CV_INTER_NN) {
		nearestNeighbor(src, dest, xscale, yscale);
	} else if (interpolation == CV_INTER_LINEAR) {
		bilinear(src, dest, xscale, yscale);
	}
}


void euclideanTransform(const Mat &src, Mat &dest, double theta, double tx, double ty) {

	double rad = theta * (M_PI / 180);

	dest = Mat::zeros(src.rows * 2, src.cols * 2, CV_8UC3);
	Mat srcCpy;
	copyMakeBorder(src, srcCpy, 1, 1, 1, 1, BORDER_REFLECT_101);

	double data[3][3] = {{cos(rad), -sin(rad), tx},
	                     {sin(rad), cos(rad),  -ty},
	                     {0,          0,           1}};


	Mat rotateMat(3, 3, CV_64FC1, data);

	for(int y = 0; y < rotateMat.rows; ++y) {
		for(int x = 0; x < rotateMat.cols; ++x) {
			cout << rotateMat.ptr<double>(y)[x]	 << endl;
		}
	}

	Mat invMat = rotateMat;

	for (int y = 0; y < dest.rows; ++y) {
		Vec3b *destTmp = dest.ptr<Vec3b>(y);
		for (int x = 0; x < dest.cols; ++x) {

			//destの中心を0,0とする
			double pos[3] = {x - dest.cols / 2.0, y - dest.rows / 2.0, 1.0};

			//srcでの位置を計算
			Mat srcPos = invMat * Mat(3, 1, CV_64FC1, pos);

			double srcX = srcPos.ptr<double>(0)[0] + srcCpy.cols / 2.0 + 0.5;
			int floorX = static_cast<int>(srcX);
			double srcY = srcPos.ptr<double>(1)[0] + srcCpy.rows / 2.0 + 0.5;
			int floorY = static_cast<int>(srcY);

			if (floorX >= 0 && floorX < srcCpy.cols && floorY >= 0 && floorY < srcCpy.rows) {
				const Vec3b *srcTmp = srcCpy.ptr<Vec3b>(floorY);
				destTmp[x][0] = srcTmp[floorX][0];
				destTmp[x][1] = srcTmp[floorX][1];
				destTmp[x][2] = srcTmp[floorX][2];
			}

		}
	}

}

int main() {

	const string windowName = "Window";
	namedWindow(windowName, WINDOW_AUTOSIZE);


	Mat src = imread("./img/lenna.png");
	Mat dest;

	euclideanTransform(src, dest, 360, 0, 0);
	imshow(windowName, dest);


	while (1) {

		if (waitKey(1) == 'q') {
			break;
		}
	}

	destroyAllWindows();

	return 0;
}
