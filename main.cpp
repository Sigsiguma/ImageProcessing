#define _USE_MATH_DEFINES

#include <vector>
#include <numeric>
#include "include.hpp"
#include "sample_code.hpp"
#include "plot.hpp"

using namespace std;
using namespace cv;

void Yellow(const Mat src, const Mat srcHSV) {
	//マスク作成
	Mat mask;
	cv::inRange(srcHSV, Scalar(20, 175, 100), Scalar(40, 255, 255), mask);
	imshow("Mask", mask);

	morphologyEx(mask, mask, CV_MOP_OPEN, Mat(3, 3, CV_8U), Point(-1, -1), 2);
	morphologyEx(mask, mask, CV_MOP_CLOSE, Mat(3, 3, CV_8U), Point(-1, -1), 16);
	imshow("AfterMask", mask);

	//作成したマスクと論理積を取る
	Mat result;
	bitwise_and(src, src, result, mask);
	imshow("Reuslt", result);
}

void Orange(const Mat src, const Mat srcHSV) {
	//マスク作成
	Mat mask;
	cv::inRange(srcHSV, Scalar(5, 160, 100), Scalar(20, 255, 255), mask);

	imshow("Mask", mask);

	morphologyEx(mask, mask, CV_MOP_OPEN, Mat(3, 3, CV_8U), Point(-1, -1), 1);
	morphologyEx(mask, mask, CV_MOP_CLOSE, Mat(3, 3, CV_8U), Point(-1, -1), 10);
	imshow("AfterMask", mask);

	//作成したマスクと論理積を取る
	Mat result;
	bitwise_and(src, src, result, mask);
	imshow("Reuslt", result);
}

void Green(const Mat src, const Mat srcHSV) {
	//マスク作成
	Mat mask;
	cv::inRange(srcHSV, Scalar(40, 100, 20), Scalar(100, 255, 255), mask);

	imshow("Mask", mask);

	morphologyEx(mask, mask, CV_MOP_CLOSE, Mat(3, 3, CV_8U), Point(-1, -1), 20);
	imshow("AfterMask", mask);

	//作成したマスクと論理積を取る
	Mat result;
	bitwise_and(src, src, result, mask);
	imshow("Reuslt", result);
}

void Pink(const Mat src, const Mat srcHSV) {
	//マスク作成
	Mat mask;
	cv::inRange(srcHSV, Scalar(150, 10, 20), Scalar(180, 255, 255), mask);

	imshow("Mask", mask);

	morphologyEx(mask, mask, CV_MOP_CLOSE, Mat(3, 3, CV_8U), Point(-1, -1), 20);
	imshow("AfterMask", mask);

	//作成したマスクと論理積を取る
	Mat result;
	bitwise_and(src, src, result, mask);
	imshow("Reuslt", result);
}

void Blue(const Mat src, const Mat srcHSV) {
	//マスク作成
	Mat mask;
	cv::inRange(srcHSV, Scalar(112, 20, 0), Scalar(125, 255, 255), mask);

	imshow("Mask", mask);

	morphologyEx(mask, mask, CV_MOP_CLOSE, Mat(3, 3, CV_8U), Point(-1, -1), 20);
	imshow("AfterMask", mask);

	//作成したマスクと論理積を取る
	Mat result;
	bitwise_and(src, src, result, mask);
	imshow("Reuslt", result);
}

int main() {

	const string windowName = "Window";
	namedWindow(windowName, WINDOW_AUTOSIZE);

	Mat src = imread("./img/Kodak/kodim03.png");
	imshow("Window", src);
	Mat srcHSV;
	cvtColor(src, srcHSV, CV_BGR2HSV);

//	Orange(src, srcHSV);
//	Yellow(src, srcHSV);
//	Green(src, srcHSV);
//	Pink(src, srcHSV);
	Blue(src, srcHSV);

	while (1) {
		if (waitKey(1) == 'q') {
			break;
		}
	}

	destroyAllWindows();

	return 0;
}
