#pragma once

//第2回課題用
void alphaBlend(const cv::Mat src1_, const cv::Mat src2_, cv::Mat& dest_, float alpha);

//第3回課題用
std::vector<uchar> lutNegativePositive();
std::vector<uchar> lutPosterization(int N);

//第4回課題用
void addGaussianNoise(const cv::Mat src, cv::Mat &dst, double sigma);
void addSpikeNoise(const cv::Mat src, cv::Mat& dest, double noise_rate);
