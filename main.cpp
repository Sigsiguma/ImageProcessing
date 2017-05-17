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

uchar filter(const Mat &src, const vector<double> &kernel, int srcX, int srcY, int row, int col, int radius) {

    double result = 0;

    for (int y = 0; y < row; ++y) {
        const uchar *imgSrc = src.ptr<uchar>(srcY + y - radius);
        for (int x = 0; x < col; ++x) {
            result += imgSrc[srcX + x - radius] * kernel.at(y * col + x);
        }
    }

    return clamp(static_cast<int>(result), 0, 255);
}

vector<double> createGaussianKernel(double sigma) {

    vector<double> data;

    int radius = sigma * 3;
    double sigma2 = sigma * sigma;

    for (int y = -radius; y <= radius; ++y) {
        for (int x = -radius; x <= radius; ++x) {
            data.emplace_back((1.0 / (2.0 * M_PI * sigma2)) *
                              exp(-(x * x + y * y) / (2.0 * sigma2)));
        }
    }

    return data;
}

double pixelValueWeight(const Mat &src, int x, int y) {

    CV_Assert(src.type() == CV_8UC1);
}

vector<double> createBilateralKernel(const Mat &src, double sigma) {

    CV_Assert(src.type() == CV_8UC1);

    vector<double> data;

    int radius = sigma * 3;
    double sigma2 = sigma * sigma;

    for (int y = -radius; y <= radius; ++y) {
        for (int x = -radius; x <= radius; ++x) {

        }
    }

}

void gaussianFilter(const Mat &src, Mat &dest, double sigma) {

    CV_Assert(src.size() == dest.size());
    CV_Assert(src.type() == CV_8UC1);
    CV_Assert(dest.type() == CV_8UC1);

    vector<double> data = createGaussianKernel(sigma);

    int radius = 3 * sigma;

    for (int y = 0; y < src.rows; ++y) {
        const uchar *imgSrc = src.ptr<uchar>(y);
        uchar *imgDest = dest.ptr<uchar>(y);
        for (int x = 0; x < src.cols; ++x) {
            if (y < radius || x < radius || y > (src.rows - 1) - radius || x > (src.cols - 1) - radius) {
                imgDest[x] = imgSrc[x];
                continue;
            }

            imgDest[x] = filter(src, data, x, y, 2 * radius + 1, 2 * radius + 1, radius);
        }
    }
}

void laplacianFilter(const Mat &src, Mat &dest) {

    CV_Assert(src.size() == dest.size());
    CV_Assert(src.type() == CV_8UC1);
    CV_Assert(dest.type() == CV_8UC1);

    vector<double> data = {0, 1, 0, 1, -4, 1, 0, 1, 0};

    int radius = 1;

    for (int y = 0; y < src.rows; ++y) {
        const uchar *imgSrc = src.ptr<uchar>(y);
        uchar *imgDest = dest.ptr<uchar>(y);
        for (int x = 0; x < src.cols; ++x) {
            if (y < radius || x < radius || y > (src.rows - 1) - radius || x > (src.cols - 1) - radius) {
                imgDest[x] = imgSrc[x];
                continue;
            }

            imgDest[x] = filter(src, data, x, y, 2 * radius + 1, 2 * radius + 1, radius);
        }
    }

}

void bilateralFilter(const Mat &src, Mat &dest, double space_sigma, double color_sigma) {

    CV_Assert(src.size() == dest.size());
    CV_Assert(src.type() == CV_8UC1);
    CV_Assert(dest.type() == CV_8UC1);

}

void UnsharpMask(const Mat &src, Mat &dest, double sigma, int k) {

    CV_Assert(src.size() == dest.size());

    gaussianFilter(src, dest, sigma);

    Mat diff = dest - src;

    dest = src + k * diff;
}


int main() {

    const string windowName = "Window";
    namedWindow(windowName, WINDOW_AUTOSIZE);

    Mat img = imread("./img/lenna.png", CV_8UC1);
    Mat dest;

    img.copyTo(dest);


    gaussianFilter(img, dest, 2.0);

    while (1) {

        imshow(windowName, dest);

        if (waitKey(1) == 'q') {
            break;
        }
    }

    destroyAllWindows();

    return 0;
}

