#include "plot.hpp"

using namespace std;
using namespace cv;

Plot::Plot(string gname_, double xmin_, double xmax_, double ymin_, double ymax_)
{
	gname = gname_;

	xmin = xmin_;
	xmax = xmax_;
	ymin = ymin_;
	ymax = ymax_;

	//default
	xscale = 1;
	yscale = 1;
	plot_type = PLOT_TYPE_LINE;
	grid_level = 3;

	area_size = Size((xmax - xmin + 1) * xscale, (ymax - ymin + 1) * yscale);

	graph = Mat(Size(area_size.width + 2 * MARGIN, area_size.height + 2 * MARGIN), CV_8UC3);
	plot_area = Mat(graph, Rect(Point(MARGIN, MARGIN), Point(MARGIN + area_size.width, MARGIN + area_size.height)));

	plotGraphBase(grid_level);
}

void Plot::resize(double xmin_, double xmax_, double ymin_, double ymax_)
{
	xmin = xmin_;
	xmax = xmax_;
	ymin = ymin_;
	ymax = ymax_;
	area_size = Size((xmax - xmin + 1) * xscale, (ymax - ymin + 1) * yscale);

	graph = Mat(Size(area_size.width + 2 * MARGIN, area_size.height + 2 * MARGIN), CV_8UC3);
	plot_area = Mat(graph, Rect(Point(MARGIN, MARGIN), Point(area_size.width + MARGIN, area_size.height + MARGIN)));

	plotGraphBase(grid_level);
	plotData();
}

Point Plot::coordinate(int x, int y)
{
	Point p((x - xmin) * xscale, ((plot_area.rows - 1) - (y - ymin) * yscale));
	return p;
}

Point Plot::coordinate(Point p)
{
	return coordinate(p.x, p.y);
}

void Plot::plotGraphBase(int level)
{
	string buff;
	int fontType = FONT_HERSHEY_COMPLEX;
	float fontSize = 0.5;
	Size textSize;
	int baseline = 0;

	graph.setTo(COLOR_WHITE);

	if (level > 2)
	{
		rectangle(plot_area, Point(0, 0), Point(plot_area.cols - 1, plot_area.rows - 1), COLOR_BLACK);

		//x coordinate
		buff = format("%.2f", (xmax - xmin) * 0.25 + xmin);
		textSize = getTextSize(buff, fontType, fontSize, 1, &baseline);
		putText(graph, buff, Point(MARGIN + plot_area.cols * 0.25 - textSize.width / 2, MARGIN + plot_area.rows + 2 * textSize.height), fontType, fontSize, COLOR_BLACK);
		line(plot_area, Point(plot_area.cols * 0.25, plot_area.rows), Point(plot_area.cols * 0.25, 0), COLOR_GRAY200);

		buff = format("%.2f", (xmax - xmin) * 0.5 + xmin);
		textSize = getTextSize(buff, fontType, fontSize, 1, &baseline);
		putText(graph, buff, Point(MARGIN + plot_area.cols * 0.5 - textSize.width / 2, MARGIN + plot_area.rows + textSize.height), fontType, fontSize, COLOR_BLACK);
		line(plot_area, Point(plot_area.cols * 0.5, plot_area.rows), Point(plot_area.cols * 0.5, 0), COLOR_GRAY200);

		buff = format("%.2f", (xmax - xmin) * 0.75 + xmin);
		textSize = getTextSize(buff, fontType, fontSize, 1, &baseline);
		putText(graph, buff, Point(MARGIN + plot_area.cols * 0.75 - textSize.width / 2, MARGIN + plot_area.rows + 2 * textSize.height), fontType, fontSize, COLOR_BLACK);
		line(plot_area, Point(plot_area.cols * 0.75, plot_area.rows), Point(plot_area.cols * 0.75, 0), COLOR_GRAY200);

		//y coordinate
		buff = format("%.2f", (ymax - ymin) * 0.25 + ymin);
		textSize = getTextSize(buff, fontType, fontSize, 1, &baseline);
		putText(graph, buff, Point(MARGIN - textSize.width, MARGIN + plot_area.rows * (1.0 - 0.25) + textSize.height / 2), fontType, fontSize, COLOR_BLACK);
		line(plot_area, Point(0, plot_area.rows * (1.0 - 0.25)), Point(plot_area.cols, plot_area.rows * (1.0 - 0.25)), COLOR_GRAY200);

		buff = format("%.2f", (ymax - ymin) * 0.5 + ymin);
		textSize = getTextSize(buff, fontType, fontSize, 1, &baseline);
		putText(graph, buff, Point(MARGIN - textSize.width, MARGIN + plot_area.rows * (1.0 - 0.5) + textSize.height / 2), fontType, fontSize, COLOR_BLACK);
		line(plot_area, Point(0, plot_area.rows * (1.0 - 0.5)), Point(plot_area.cols, plot_area.rows * (1.0 - 0.5)), COLOR_GRAY200);

		buff = format("%.2f", (ymax - ymin) * 0.75 + ymin);
		textSize = getTextSize(buff, fontType, fontSize, 1, &baseline);
		putText(graph, buff, Point(MARGIN - textSize.width, MARGIN + plot_area.rows * (1.0 - 0.75) + textSize.height / 2), fontType, fontSize, COLOR_BLACK);
		line(plot_area, Point(0, plot_area.rows * (1.0 - 0.75)), Point(plot_area.cols, plot_area.rows * (1.0 - 0.75)), COLOR_GRAY200);
	}
	if (level > 1)
	{
		rectangle(plot_area, Point(0, 0), Point(plot_area.cols - 1, plot_area.rows - 1), COLOR_BLACK);

		//x coordinate
		buff = format("%.2f", xmin);
		textSize = getTextSize(buff, fontType, fontSize, 1, &baseline);
		putText(graph, buff, Point(MARGIN - textSize.width / 2, graph.rows - MARGIN + textSize.height), fontType, fontSize, COLOR_BLACK);

		buff = format("%.2f", xmax);
		textSize = getTextSize(buff, fontType, fontSize, 1, &baseline);
		putText(graph, buff, Point(graph.cols - MARGIN - textSize.width / 2, graph.rows - MARGIN + textSize.height), fontType, fontSize, COLOR_BLACK);

		//y coordinate
		buff = format("%.2f", ymin);
		textSize = getTextSize(buff, fontType, fontSize, 1, &baseline);
		putText(graph, buff, Point(MARGIN - textSize.width, graph.rows - MARGIN), fontType, fontSize, COLOR_BLACK);

		buff = format("%.2f", ymax);
		textSize = getTextSize(buff, fontType, fontSize, 1, &baseline);
		putText(graph, buff, Point(MARGIN - textSize.width, MARGIN), fontType, fontSize, COLOR_BLACK);
	}
	if (level > 0)
	{
		textSize = getTextSize("X", fontType, 2 * fontSize, 1, &baseline);
		putText(graph, "X", Point(MARGIN + plot_area.cols * 0.5 - textSize.width / 2, MARGIN + plot_area.rows + 3 * textSize.height), fontType, fontSize, COLOR_BLACK);
		textSize = getTextSize("Y", fontType, 2 * fontSize, 1, &baseline);
		putText(graph, "Y", Point(0, (graph.rows + textSize.height) / 2), fontType, fontSize, COLOR_BLACK);
	}
}

void Plot::plotPoint(Point p, Scalar color)
{
	circle(plot_area, p, 1, color, -1);
}

void Plot::plotLine(Point p1, Point p2, Scalar color)
{
	line(plot_area, p1, p2, color);
}

template <typename T>
void Plot::plotData_(vector <T> data, Scalar color)
{
	if ((xmax - xmin + 1) < data.size())
	{
		xmax = xmin + data.size() - 1;
		resize(xmin, xmax, ymin, ymax);
	}

	if (plot_type == PLOT_TYPE_LINE)
		plotPoint(coordinate(0, data[0]), color);
	if (plot_type == PLOT_TYPE_AREA)
		plotLine(coordinate(0, data[0]), coordinate(0, 0), color);
	for (int i = 1; i < data.size(); i++)
	{
		if (plot_type == PLOT_TYPE_LINE)
			plotPoint(coordinate(i, data[i]), color);

		if (plot_type == PLOT_TYPE_AREA)
			plotLine(coordinate(i, data[i]), coordinate(i, 0), color);

		plotLine(coordinate(i - 1, data[i - 1]), coordinate(i, data[i]), color);
	}
}


void Plot::plotData(vector<double> data, Scalar color)
{
	Plot::plotData_<double>(data, color);
}

void Plot::plotData(vector<float> data, Scalar color)
{
	Plot::plotData_<float>(data, color);
}

void Plot::plotData(vector<int> data, Scalar color)
{
	Plot::plotData_<int>(data, color);
}

void Plot::plotData(vector<uchar> data, Scalar color)
{
	Plot::plotData_<uchar>(data, color);
}

template <typename T>
void Plot::plotData_(T* data_, int data_size, cv::Scalar color)
{
	vector<double> data;

	for (int i = 0; i < data_size; i++)
		data.push_back(data_[i]);

	Plot::plotData_<double>(data, color);
}
void Plot::plotData(double* data, int data_size, cv::Scalar color)
{
	Plot::plotData_<double>(data, data_size, color);
}
void Plot::plotData(float* data, int data_size, cv::Scalar color)
{
	Plot::plotData_<float>(data, data_size, color);
}
void Plot::plotData(int* data, int data_size, cv::Scalar color)
{
	Plot::plotData_<int>(data, data_size, color);
}
void Plot::plotData(uchar* data, int data_size, cv::Scalar color)
{
	Plot::plotData_<uchar>(data, data_size, color);
}

void Plot::plotData()
{
	for (int i = 0; i < data_list.size(); i++)
		plotData(data_list[i], color_list[i]);
}

void Plot::seGridLevel(int grid_level_)
{
	grid_level = grid_level_;
}

void Plot::setPlotType(int Plot_type_)
{
	plot_type = Plot_type_;
}

void Plot::setScale(double xscale_, double yscale_)
{
	if ((xscale != xscale_) || (yscale != yscale_))
	{
		if (xscale_ > 0)
			xscale = xscale_;
		if (yscale_ > 0)
			yscale = yscale_;

		resize(xmin, xmax, ymin, ymax);
	}
}

template <typename T>
void Plot::setData_(vector<T> data_)
{
	vector<double> data = vector<double>(data_.begin(), data_.end());
	data_list.push_back(data);
}
void Plot::setData(vector<double> data)
{
	setData_<double>(data);
}
void Plot::setData(vector<float> data)
{
	setData_<float>(data);
}
void Plot::setData(vector<int> data)
{
	setData_<int>(data);
}
void Plot::setData(vector<uchar> data)
{
	setData_<uchar>(data);
}

template <typename T>
void Plot::setData_(T* data_, int data_size)
{
	vector<double> data;

	for (int i = 0; i < data_size; i++)
		data.push_back(data_[i]);

	data_list.push_back(data);
}
void Plot::setData(double* data_, int data_size)
{
	setData_<double>(data_, data_size);
}
void Plot::setData(float* data_, int data_size)
{
	setData_<float>(data_, data_size);
}
void Plot::setData(int* data_, int data_size)
{
	setData_<int>(data_, data_size);
}
void Plot::setData(uchar* data_, int data_size)
{
	setData_<uchar>(data_, data_size);
}

void Plot::imshow()
{
	plotData();
	cv::imshow(gname, graph);
}

void histogram(const Mat img_)
{
	CV_Assert(img_.depth() == CV_8U && img_.channels() == 1);

	Mat img = img_.clone();

	vector<double> hist(256);

	const uchar* ptr = img.ptr<uchar>(0);
	for (int i = 0; i < img.size().area(); i++)
	{
		hist[*(ptr++)]++;
	}

	Plot p("histogram", 0, 255, 0, *max_element(hist.begin(), hist.end()));
	p.setScale(1, 255 / *max_element(hist.begin(), hist.end()));
	p.setPlotType(Plot::PLOT_TYPE_AREA);
	p.plotData(hist, COLOR_BLACK);
	p.imshow();
}

void tone_curve(const vector<uchar> lut)
{
	Plot p("tone_curve");
	p.plotData(lut, COLOR_BLACK);
	p.imshow();
}