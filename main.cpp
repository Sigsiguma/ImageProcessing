#define _USE_MATH_DEFINES

#include <vector>
#include <numeric>
#include "include.hpp"
#include "sample_code.hpp"
#include "plot.hpp"

using namespace std;
using namespace cv;

int main() {

	const string windowName = "Window";
	namedWindow(windowName, WINDOW_AUTOSIZE);

	Mat src = imread("./img/Kodak/kodim03.png");
	imshow("Window", src);
	Mat srcHSV;
	cvtColor(src, srcHSV, CV_BGR2HSV);

	//マスク作成
	Mat mask;
	cv::inRange(srcHSV, Scalar(16, 175, 100), Scalar(38, 255, 255), mask);
	imshow("Mask", mask);

	morphologyEx(mask, mask, CV_MOP_CLOSE, Mat(3, 3, CV_8U), Point(-1, -1), 10);
	imshow("AfterMask", mask);

	//作成したマスクと論理積を取る
	Mat result;
	bitwise_and(src, src, result, mask);

	imshow("Reuslt", result);


	while (1) {
		if (waitKey(1) == 'q') {
			break;
		}
	}

	destroyAllWindows();

	return 0;
}
