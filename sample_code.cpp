#include "include.hpp"
#include "sample_code.hpp"
#include <random>

//アルファブレンド
void alphaBlend(const cv::Mat src1_, const cv::Mat src2_, cv::Mat& dest_, float alpha)
{
	CV_Assert(src1_.size() == src2_.size());

	cv::Mat src1, src2, dest;
	src1_.convertTo(src1, CV_32F);
	src2_.convertTo(src2, CV_32F);

	dest = alpha * src1 + (1.0f - alpha) *src2;

	dest.convertTo(dest_, CV_8U);
}

std::vector<uchar> lutNegativePositive()
{
	std::vector<uchar> table(256);

	for (int i = 0; i < 256; i++)
		table[i] = 255 - i;

	return table;
}

std::vector<uchar> lutPosterization(int N)
{
	std::vector<uchar> table(256);

	for (int i = 0; i < 256; i++)
		table[i] = uchar(i * N / 256) * (255.0 / double(N - 1));

	return table;
}

// ノイズ付加 (1ch)
void addGaussianNoiseMono(const cv::Mat src, cv::Mat &dst, double sigma)
{
	cv::Mat s;
	src.convertTo(s, CV_16S);
	cv::Mat n(s.size(), CV_16S);
	cv::randn(n, 0, sigma);
	cv::Mat temp = s + n;
	temp.convertTo(dst, CV_8U);
}
// ノイズ付加 (3ch)
void addGaussianNoise(const cv::Mat src, cv::Mat &dest, double sigma)
{
	std::vector<cv::Mat> s;
	std::vector<cv::Mat> d(src.channels());
	cv::split(src, s);
	for (int i = 0; i < src.channels(); i++)
	{
		addGaussianNoiseMono(s[i], d[i], sigma);
	}
	cv::merge(d, dest);
}

void addSpikeNoise(const cv::Mat src, cv::Mat& dest, double noise_rate)
{
	dest = src.clone();

	if (noise_rate > 100)
	{
		std::cout << " noise_rate 0~100" << std::endl;
		return;
	}
	std::random_device rnd;     // 非決定的な乱数生成器
	std::mt19937 mt(rnd());            // メルセンヌ・ツイスタの32ビット版、引数は初期シード

	int data_size = src.size().area();

	int p;
	for (int i = 0; i < data_size; i++)
	{
		if (double(mt() % 10000) / 100.0 < noise_rate)
		{
			if (dest.channels() == 3)
			{
				dest.ptr<cv::Vec3b>(0)[i] = cv::Vec3b(255, 255, 255) * int(mt() % 2);
			}
			else
			{
				dest.ptr<uchar>(0)[i] = 255 * (rand() % 2);
			}
		}
	}

}