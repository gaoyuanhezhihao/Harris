#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <limits>

#define ALPHA 0.04
using namespace cv;
using namespace std;

/// Global variables
Mat src, src_gray, src_gray_8u;
int thresh = 1;
int max_thresh = 255;

char* source_window = "Source image";
char* corners_window = "Corners detected";
// Variable
enum return_code{
	success,
	too_few_point,
};
/// Function header
void cornerHarris_demo(int thres, int point_num, bool do_supress);
return_code _suppress(vector<pair<double, Point>> &harris_points, vector<pair<double, Point>> &radius_harris, int n);
double _get_min_radius(vector<pair<double, Point>> harris_points, double reaction, Point coordination);
int main(int argc, char** argv)
{
	/// Load source image and convert it to gray
	int thres;
	int point_num;
	bool do_supress = false;
	if (argc < 5+1)
	{
		cout << "Error: too few param" << '\n' << "@1. input image name\n@2. output file name \n"
			<<"@3.thres\n@ 4. number of points\n @5. make supress or not"
			<< "example: chessboard.png chessboard_harris.png 100 50 y"
			<< endl;
		exit(-1);
	}
	if (argv[5][0] == 'y')
	{
		do_supress = true;
	}
	else
	{
		do_supress = false;
	}
	point_num = atoi(argv[4]);
	thres = atoi(argv[3]);
	src = imread(argv[1], 1);
	cvtColor(src, src_gray_8u, CV_BGR2GRAY);
	src_gray_8u.convertTo(src_gray, CV_64FC1);
	/// Create a window and a trackbar
	namedWindow(source_window, CV_WINDOW_AUTOSIZE);
	imshow(source_window, src);

	cornerHarris_demo(thres, point_num, do_supress);
	imwrite(argv[2], src_gray);
	waitKey(0);
	return(0);
}

void cornerHarris_demo(int thres, int n, bool do_supress)
{
	struct sort_match{
		bool operator()(const pair<double, Point > & left, pair<double, Point > & right)
		{
			return left.first > right.first;
		}
	};
	Mat Ix_kernel = (Mat_<double>(3, 3) << -1, 0, 1, -1, 0, 1, -1, 0, 1);
	Mat Iy_kernel = (Mat_<double>(3, 3) << -1, -1, -1, 0, 0, 0, 1, 1, 1);
	Mat im_Ix(src_gray.size(), CV_64FC1);
	Mat im_Iy(src_gray.size(), CV_64FC1);
	Mat IxIx(src_gray.size(), CV_64FC1);
	Mat IyIy(src_gray.size(), CV_64FC1);
	Mat IxIy(src_gray.size(), CV_64FC1);
	Mat gIxIx(src_gray.size(), CV_64FC1);
	Mat gIxIy(src_gray.size(), CV_64FC1);
	Mat gIyIy(src_gray.size(), CV_64FC1);
	Mat R_mat(src_gray.size(), CV_64FC1);
	//Mat H_mat(2, 2, CV_64FC1);
	vector<pair<double, Point>> harris_points;
	vector<pair<double, Point>> radius_harris;
	int i, j;
	Mat lambda, eigen_vec;
	double trace_tmp;
	double R = 0;

	Ix_kernel.at<double>(0, 0) = -1.0;
	Ix_kernel.at<double>(0, 1) = 2;
	Ix_kernel.at<double>(0, 2) = -1.0;
	Iy_kernel.at<double>(0, 0) = -1.0;
	Iy_kernel.at<double>(1, 0) = 2;
	Iy_kernel.at<double>(2, 0) = -1.0;
	blur(src_gray, src_gray, Size(3, 3));
	filter2D(src_gray, im_Ix, -1, Ix_kernel, Point(-1, -1), 0, BORDER_DEFAULT);
	filter2D(src_gray, im_Iy, -1, Iy_kernel, Point(-1, -1), 0, BORDER_DEFAULT);
	//Sobel(src_gray, im_Ix, CV_64FC1, 1, 0, 3, BORDER_DEFAULT);
	//Sobel(src_gray, im_Iy, CV_64FC1, 0, 1, 3, BORDER_DEFAULT);
	pow(im_Ix, 2.0, IxIx);
	pow(im_Iy, 2.0, IyIy);
	multiply(im_Ix, im_Iy, IxIy);

	GaussianBlur(IxIx, gIxIx, Size(7, 7), 2.0, 0.0, BORDER_DEFAULT);
	GaussianBlur(IyIy, gIyIy, Size(7, 7), 0.0, 2.0, BORDER_DEFAULT);
	GaussianBlur(IxIy, gIxIy, Size(7, 7), 2.0, 0.0, BORDER_DEFAULT);

	//blur(IyIy, IyIy, Size(3, 3));
	//blur(IxIy, IxIy, Size(3, 3));
	double * p_IxIx, * p_IyIy, *p_IxIy;
	MatIterator_<double> it_IxIx=gIxIx.begin<double>(), it_IxIy= gIxIy.begin<double>(), it_IyIy=gIyIy.begin<double>();
	int nRows = IxIy.rows;
	int nCols = IxIy.cols;
	double debug_max = 0;
	double det = 0;
	double trac = 0;
	for (i=0; i< nRows; ++i)
	{
		p_IxIx = IxIx.ptr<double>(i);
		p_IxIy = IxIy.ptr<double>(i);
		p_IyIy = IyIy.ptr<double>(i);
		for (j = 0; j < nCols; ++j)
		{
			//Mat H_mat = (Mat_<double>(2, 2) << *it_IxIx++,*it_IxIy, *it_IxIy++ , *it_IyIy++);
			//Mat H_mat(2, 2, CV_64FC1);
			//H_mat.at<double>(0, 0) = *(it_IxIx++);
			//H_mat.at<double>(0, 1) = *it_IxIy;
			//H_mat.at<double>(1, 0) = *(it_IxIy++);
			//H_mat.at<double>(1, 1) = *(it_IyIy++);
			////cout << H_mat << endl;
			//PCA h_pca(H_mat, Mat(), CV_PCA_DATA_AS_ROW, 0);
			//lambda = h_pca.eigenvalues;
			//R = lambda.at<double>(0) * lambda.at<double>(1) / (lambda.at<double>(0) + lambda.at<double>(1));
			
			R_mat.at<double>(i, j) = R;
			det = *it_IxIx * *it_IyIy - *it_IxIy * *it_IxIy;
			trac = *it_IxIx + *it_IyIy;
			R = det / trac;
			++it_IxIx;
			++it_IxIy;
			++it_IyIy;
			if (R > debug_max)
			{
				debug_max = R;
			}

			//cout << H_mat.at<double>(0, 0) << endl;
			if (R > thres)
			{
				harris_points.push_back(  pair<double, Point>( R, Point(j, i) )  );
			}
		}
	}
	if (do_supress)
	{
		if (_suppress(harris_points, radius_harris, n) == return_code::too_few_point)
		{
			cout << "less than " << n << " points founded" << endl;
			auto it = harris_points.begin();
			for (; it != harris_points.end(); ++it)
			{
				circle(src_gray, it->second, 5, Scalar(0), 2, 8, 0);
			}
		}
		else
		{
			cout << "got " << n << " points" << endl;
			auto it = radius_harris.begin();
			int k = 0;
			for (k = 0; it != radius_harris.end() && k < n; ++it, ++k)
			{
				circle(src_gray, it->second, 5, Scalar(0), 2, 8, 0);
				//cout << it->first << ", " << it->second << endl;
			}
		}
	}
	else
	{
		sort(harris_points.begin(), harris_points.end(), sort_match());
		auto it = harris_points.begin();
		int k = 0;
		for (; it != harris_points.end() && k < n; ++it, ++k)
		{
			circle(src_gray, it->second, 5, Scalar(0), 2, 8, 0);
		}
	}


	cout << "size of element:" << sizeof(IyIy.at<double>(0,0)) << endl;
	cout << "r_max:" << debug_max << endl;
	cout << "nRow:" << nRows << ", nCols:" << nCols << endl;
	//for (i = 0; i < src_gray.size().height; i++)
	//{
	//	for (j = 0; j < src_gray.size().width; j++)
	//	{
	//		H_mat.at<double
	//	}
	//}

	//Mat m1(2, 2, CV_64FC1);
	//m1.at<double>(0, 0) = 1;
	//m1.at<double>(0, 1) = 2;
	//m1.at<double>(1, 0) = 3;
	//m1.at<double>(1, 1) = 4;

	//Mat m2(2, 2, CV_64FC1);
	//m2 = m1;
	//cout << m1.mul(m2) << endl;
	//imshow("IxIx", IxIx);
	//imshow("IyIy", IyIy);
	//imshow("IxIy", IxIy);
	//imshow("R_mat", R_mat);
	imshow("harris corner", src_gray);
	int c = waitKey(1);
}

return_code _suppress(vector<pair<double, Point>> &harris_points, vector<pair<double, Point>> &radius_harris, int n)
{
	struct sort_match{
		bool operator()(const pair<double, Point > & left, pair<double, Point > & right)
		{
			return left.first > right.first;
		}
	};
	if (harris_points.size() < n)
	{
		return return_code::too_few_point;
	}
	sort(harris_points.begin(), harris_points.end(), sort_match());
	auto it = harris_points.begin();
	double min_radius = 0;
	for (; it != harris_points.end(); ++it)
	{
		min_radius = _get_min_radius(harris_points, it->first, it->second);
		radius_harris.push_back(pair<double, Point>(min_radius, it->second));
	}
	sort(radius_harris.begin(), radius_harris.end(), sort_match());
	return return_code::success;
}

double _get_min_radius(vector<pair<double, Point>> harris_points, double reaction, Point pt)
{
	auto it = harris_points.begin();
	double min_radius = std::numeric_limits<double>::max();
	double thres = reaction;
	double tmp_radius = 0;
	for (; it != harris_points.end(); ++it)
	{
		if (it->first <= thres) 
		{	// iterate over
			return min_radius;
		}
		tmp_radius = norm(it->second - pt);
		if (tmp_radius < min_radius)
		{
			min_radius = tmp_radius;
		}
	}
	return min_radius;
}