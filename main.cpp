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

void gaussianFilter(const Mat &src, Mat &dest, double sigma) {

}

void laplacianFilter(const Mat &src, Mat &dest) {

    float data[3][3] = {{0, 1,  0},
                        {1, -4, 1},
                        {0, 1,  0}};

    Mat kernel(Size(3, 3), CV_32F, data);

    filter2D(src, dest, -1, kernel);
}

void bilateralFilter(const Mat &src, Mat &dest, double space_sigma, double color_sigma) {

}

int main() {

    const string windowName = "Window";
    namedWindow(windowName, WINDOW_AUTOSIZE);

    Mat img = imread("./img/lenna.png");
    Mat gray;
    Mat dest;

    cvtColor(img, gray, CV_BGR2GRAY);
    laplacianFilter(gray, dest);

    while (1) {

        imshow(windowName, dest);

        if (waitKey(1) == 'q') {
            break;
        }
    }

    destroyAllWindows();

    return 0;
}

