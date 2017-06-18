#define _USE_MATH_DEFINES

#include <vector>
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

	morphologyEx(mask, mask, MORPH_OPEN, Mat(), Point(-1, -1), 2);
	morphologyEx(mask, mask, MORPH_CLOSE, Mat(), Point(-1, -1), 16);
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

	morphologyEx(mask, mask, MORPH_OPEN, Mat(), Point(-1, -1), 1);
	morphologyEx(mask, mask, MORPH_CLOSE, Mat(), Point(-1, -1), 10);
	imshow("AfterMask", mask);

	//作成したマスクと論理積を取る
	Mat result;
	bitwise_and(src, src, result, mask);
	imshow("Reuslt", result);
}

void Green(const Mat src, const Mat srcHSV) {
	//マスク作成
	Mat mask;
	cv::inRange(srcHSV, Scalar(37, 68, 0), Scalar(57, 255, 255), mask);

	imshow("Mask", mask);

	morphologyEx(mask, mask, MORPH_OPEN, Mat(), Point(-1, -1), 3);
	morphologyEx(mask, mask, MORPH_CLOSE, Mat(), Point(-1, -1), 20);
	imshow("AfterMask", mask);

	//作成したマスクと論理積を取る
	Mat result;
	bitwise_and(src, src, result, mask);
	imshow("Reuslt", result);
}

void Pink(const Mat src, const Mat srcHSV) {
	//マスク作成
	Mat mask;
	cv::inRange(srcHSV, Scalar(4, 0, 0), Scalar(167, 255, 255), mask);

	mask = ~mask;
	imshow("Mask", mask);

	morphologyEx(mask, mask, MORPH_OPEN, Mat(), Point(-1, -1), 2);
	morphologyEx(mask, mask, MORPH_CLOSE, Mat(), Point(-1, -1), 10);
	imshow("AfterMask", mask);

	//作成したマスクと論理積を取る
	Mat result;
	bitwise_and(src, src, result, mask);
	imshow("Reuslt", result);
}

void Blue(const Mat src, const Mat srcHSV) {
	//マスク作成
	Mat mask;
	cv::inRange(srcHSV, Scalar(110, 40, 0), Scalar(180, 147, 137), mask);

	imshow("Mask", mask);

	morphologyEx(mask, mask, MORPH_OPEN, Mat(), Point(-1, -1), 1);
	morphologyEx(mask, mask, MORPH_CLOSE, Mat(), Point(-1, -1), 20);
	imshow("AfterMask", mask);

	//作成したマスクと論理積を取る
	Mat result;
	bitwise_and(src, src, result, mask);
	imshow("Reuslt", result);
}

enum class Color {
	Orange,
	Yellow,
	Green,
	Pink,
	Blue
};

int main() {

	const string windowName = "Window";
	namedWindow(windowName, WINDOW_AUTOSIZE);

	Mat src = imread("./img/Kodak/kodim03.png");
	imshow("Window", src);
	Mat srcHSV;
	cvtColor(src, srcHSV, CV_BGR2HSV);

	int type;
	cout << "0: Orange, 1: Yellow, 2: Green, 3: Pink, 4: Blue" << endl;
	cin >> type;

	switch (static_cast<Color>(type)) {
		case Color::Orange:
			Orange(src, srcHSV);
			break;
		case Color::Yellow:
			Yellow(src, srcHSV);
			break;
		case Color::Green:
			Green(src, srcHSV);
			break;
		case Color::Pink:
			Pink(src, srcHSV);
			break;
		case Color::Blue:
			Blue(src, srcHSV);
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
