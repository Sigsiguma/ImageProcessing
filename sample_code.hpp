#pragma once

#define IMAGE_PATH "../img/"
#define KODAK_IMAGE(id) IMAGE_PATH"Kodak/kodim" CVAUX_STR(id)".png"
#define HIGHRESOLUTIONÅQIMAGE(name)"highResolution/" CVAUX_STR(name)".png"

void alphaBlend(const cv::Mat src1_, const cv::Mat src2_, cv::Mat& dest_, float alpha);

std::vector<uchar> lutNegativePositive();
std::vector<uchar> lutPosterization(int N);

void addGaussianNoise(const cv::Mat src, cv::Mat &dst, double sigma);
void addSpikeNoise(const cv::Mat src, cv::Mat& dest, double noise_rate);

void getGaussianMaskDCT(cv::Mat& MMask, cv::Size size, double sigma);