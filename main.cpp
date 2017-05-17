#include <vector>
#include "include.hpp"
#include "sample_code.hpp"
#include "plot.hpp"

using namespace std;
using namespace cv;

vector<float> CreateGaussianKernel(double sigma) {

    vector<float> data;

    int radius = sigma * 3;
    double sigma2 = sigma * sigma;

    for (int y = -radius; y <= radius; ++y) {
        for (int x = -radius; x <= radius; ++x) {
            data.emplace_back((1.0f / (2.0f * M_PI * sigma2)) *
                              exp(-(x * x + y * y) / (2.0f * sigma2)));
        }
    }

    return data;
}

void gaussianFilter(const Mat &src, Mat &dest, double sigma) {

    CV_Assert(src.size() == dest.size());

    vector<float> data = CreateGaussianKernel(sigma);

    int w = 3 * sigma;

    Mat kernel(Size(2 * w + 1, 2 * w + 1), CV_32F, data.data());

    filter2D(src, dest, -1, kernel);
}

void laplacianFilter(const Mat &src, Mat &dest) {

    CV_Assert(src.size() == dest.size());

    float data[3][3] = {{0, 1,  0},
                        {1, -4, 1},
                        {0, 1,  0}};

    Mat kernel(Size(3, 3), CV_32F, data);

    filter2D(src, dest, -1, kernel);
}

void bilateralFilter(const Mat &src, Mat &dest, double space_sigma, double color_sigma) {

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

    Mat img = imread("./img/lenna.png");
    Mat gray;
    Mat dest;

    img.copyTo(gray);
    img.copyTo(dest);

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

