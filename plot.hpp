#pragma once
#include <iostream>
#include <fstream>
#include "include.hpp"

#define COLOR_BLACK cv::Scalar(0,0,0)
#define COLOR_GRAY10 cv::Scalar(10,10,10)
#define COLOR_GRAY20 cv::Scalar(20,20,20)
#define COLOR_GRAY30 cv::Scalar(10,30,30)
#define COLOR_GRAY40 cv::Scalar(40,40,40)
#define COLOR_GRAY50 cv::Scalar(50,50,50)
#define COLOR_GRAY60 cv::Scalar(60,60,60)
#define COLOR_GRAY70 cv::Scalar(70,70,70)
#define COLOR_GRAY80 cv::Scalar(80,80,80)
#define COLOR_GRAY90 cv::Scalar(90,90,90)
#define COLOR_GRAY100 cv::Scalar(100,100,100)
#define COLOR_GRAY110 cv::Scalar(101,110,110)
#define COLOR_GRAY120 cv::Scalar(120,120,120)
#define COLOR_GRAY130 cv::Scalar(130,130,140)
#define COLOR_GRAY140 cv::Scalar(140,140,140)
#define COLOR_GRAY150 cv::Scalar(150,150,150)
#define COLOR_GRAY160 cv::Scalar(160,160,160)
#define COLOR_GRAY170 cv::Scalar(170,170,170)
#define COLOR_GRAY180 cv::Scalar(180,180,180)
#define COLOR_GRAY190 cv::Scalar(190,190,190)
#define COLOR_GRAY200 cv::Scalar(200,200,200)
#define COLOR_GRAY210 cv::Scalar(210,210,210)
#define COLOR_GRAY220 cv::Scalar(220,220,220)
#define COLOR_GRAY230 cv::Scalar(230,230,230)
#define COLOR_GRAY240 cv::Scalar(240,240,240)
#define COLOR_GRAY250 cv::Scalar(250,250,250)
#define COLOR_WHITE cv::Scalar(255,255,255)

#define COLOR_RED cv::Scalar(0,0,255)
#define COLOR_GREEN cv::Scalar(0,255,0)
#define COLOR_BLUE cv::Scalar(255,0,0)
#define COLOR_ORANGE cv::Scalar(0,100,255)
#define COLOR_YELLOW cv::Scalar(0,255,255)
#define COLOR_MAGENDA cv::Scalar(255,0,255)
#define COLOR_CYAN cv::Scalar(255,255,0)

#define MARGIN 75

class Plot {
protected:
	std::string gname;
	cv::Mat graph;
	int grid_level;

	cv::Mat plot_area;
	double xscale;
	double yscale;
	int plot_type;

	double xmin;
	double xmax;
	double ymin;
	double ymax;
	cv::Size area_size;

	std::vector<std::vector<double>> data_list;
	std::vector<cv::Scalar> color_list =
	{ COLOR_RED,
		COLOR_GREEN,
		COLOR_BLUE,
		COLOR_ORANGE,
		COLOR_YELLOW,
		COLOR_MAGENDA,
		COLOR_CYAN };

public:
	enum
	{
		PLOT_TYPE_LINE,
		PLOT_TYPE_AREA,
	};

	Plot(std::string gname_ = "garaph", double xmin = 0, double xmax = 255, double ymin = 0, double ymax = 255);
	~Plot() {}

	void resize(double xmin, double xmax, double ymin, double ymax);

	cv::Point coordinate(int x, int y);
	cv::Point coordinate(cv::Point p);

	void plotGraphBase(int level);
	void plotPoint(cv::Point p, cv::Scalar color);
	void plotLine(cv::Point p1, cv::Point p2, cv::Scalar color);
	template <typename T>
	void plotData_(std::vector<T> data, cv::Scalar color = COLOR_BLACK);
	void plotData(std::vector<double> data, cv::Scalar color = COLOR_BLACK);
	void plotData(std::vector<float> data, cv::Scalar color = COLOR_BLACK);
	void plotData(std::vector<int> data, cv::Scalar color = COLOR_BLACK);
	void plotData(std::vector<uchar> data, cv::Scalar color = COLOR_BLACK);
	template <typename T>
	void plotData_(T* data, int data_size, cv::Scalar color = COLOR_BLACK);
	void plotData(double* data, int data_size, cv::Scalar color = COLOR_BLACK);
	void plotData(float* data, int data_size, cv::Scalar color = COLOR_BLACK);
	void plotData(int* data, int data_size, cv::Scalar color = COLOR_BLACK);
	void plotData(uchar* data, int data_size, cv::Scalar color = COLOR_BLACK);
	void plotData();

	void seGridLevel(int grid_level = 3);
	void setPlotType(int Plot_type = PLOT_TYPE_LINE);
	void setScale(double xscale = 0, double yscale = 0);

	template <typename T>
	void setData_(std::vector<T> data);
	void setData(std::vector<double> data);
	void setData(std::vector<float> data);
	void setData(std::vector<int> data);
	void setData(std::vector<uchar> data);

	template <typename T>
	void setData_(T* data, int data_size);
	void setData(double* data, int data_size);
	void setData(float* data, int data_size);
	void setData(int* data, int data_size);
	void setData(uchar* data, int data_size);

	void imshow();
};

void histogram(const cv::Mat img);
void tone_curve(const std::vector<uchar> lut);