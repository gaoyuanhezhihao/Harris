#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#define ALPHA 0.04
using namespace cv;
using namespace std;

/// Global variables
Mat src, src_gray;
int thresh = 1;
int max_thresh = 255;

char* source_window = "Source image";
char* corners_window = "Corners detected";

/// Function header
void cornerHarris_demo(int thres);

int main(int argc, char** argv)
{
	/// Load source image and convert it to gray
	int thres;
	thres = atoi(argv[2]);
	src = imread(argv[1], 1);
	cvtColor(src, src_gray, CV_BGR2GRAY);

	/// Create a window and a trackbar
	namedWindow(source_window, CV_WINDOW_AUTOSIZE);
	imshow(source_window, src);

	cornerHarris_demo(thres);

	waitKey(0);
	return(0);
}

void cornerHarris_demo(int thres)
{
	Mat Ix_kernel = (Mat_<float>(3, 3) << -1, 0, 1, -1, 0, 1, -1, 0, 1);
	Mat Iy_kernel = (Mat_<float>(3, 3) << -1, -1, -1, 0, 0, 0, 1, 1, 1);
	Mat im_Ix(src_gray.size(), CV_32FC1);
	Mat im_Iy(src_gray.size(), CV_32FC1);
	Mat IxIx(src_gray.size(), CV_32FC1);
	Mat IyIy(src_gray.size(), CV_32FC1);
	Mat IxIy(src_gray.size(), CV_32FC1);
	Mat gIxIx(src_gray.size(), CV_32FC1);
	Mat gIxIy(src_gray.size(), CV_32FC1);
	Mat gIyIy(src_gray.size(), CV_32FC1);
	Mat R_mat(src_gray.size(), CV_32FC1);
	//Mat H_mat(2, 2, CV_32FC1);
	std::vector<Point> harris_points;
	int i, j;
	Mat lambda, eigen_vec;
	double trace_tmp;
	double R = 0;

	//Ix_kernel.at<float>(0, 0) = -1.0;
	//Ix_kernel.at<float>(0, 1) = 2;
	//Ix_kernel.at<float>(0, 2) = -1.0;
	//Iy_kernel.at<float>(0, 0) = -1.0;
	//Iy_kernel.at<float>(1, 0) = 2;
	//Iy_kernel.at<float>(2, 0) = -1.0;
	blur(src_gray, src_gray, Size(3, 3));
	//filter2D(src_gray, im_Ix, -1, Ix_kernel, Point(-1, -1), 0, BORDER_DEFAULT);
	//filter2D(src_gray, im_Iy, -1, Iy_kernel, Point(-1, -1), 0, BORDER_DEFAULT);
	Sobel(src_gray, im_Ix, CV_32FC1, 1, 0, 3, BORDER_DEFAULT);
	Sobel(src_gray, im_Iy, CV_32FC1, 0, 1, 3, BORDER_DEFAULT);
	pow(im_Ix, 2.0, IxIx);
	pow(im_Iy, 2.0, IyIy);
	multiply(im_Ix, im_Iy, IxIy);

	GaussianBlur(IxIx, gIxIx, Size(7, 7), 2.0, 0.0, BORDER_DEFAULT);
	GaussianBlur(IyIy, gIyIy, Size(7, 7), 0.0, 2.0, BORDER_DEFAULT);
	GaussianBlur(IxIy, gIxIy, Size(7, 7), 2.0, 0.0, BORDER_DEFAULT);

	//blur(IyIy, IyIy, Size(3, 3));
	//blur(IxIy, IxIy, Size(3, 3));
	float * p_IxIx, * p_IyIy, *p_IxIy;
	MatIterator_<float> it_IxIx=gIxIx.begin<float>(), it_IxIy= gIxIy.begin<float>(), it_IyIy=gIyIy.begin<float>();
	int nRows = IxIy.rows;
	int nCols = IxIy.cols;
	double debug_max = 0;
	double det = 0;
	double trac = 0;
	for (i=0; i< nRows; ++i)
	{
		p_IxIx = IxIx.ptr<float>(i);
		p_IxIy = IxIy.ptr<float>(i);
		p_IyIy = IyIy.ptr<float>(i);
		for (j = 0; j < nCols; ++j)
		{
			//Mat H_mat = (Mat_<float>(2, 2) << *it_IxIx++,*it_IxIy, *it_IxIy++ , *it_IyIy++);
			//Mat H_mat(2, 2, CV_32FC1);
			//H_mat.at<float>(0, 0) = *(it_IxIx++);
			//H_mat.at<float>(0, 1) = *it_IxIy;
			//H_mat.at<float>(1, 0) = *(it_IxIy++);
			//H_mat.at<float>(1, 1) = *(it_IyIy++);
			////cout << H_mat << endl;
			//PCA h_pca(H_mat, Mat(), CV_PCA_DATA_AS_ROW, 0);
			//lambda = h_pca.eigenvalues;
			//R = lambda.at<float>(0) * lambda.at<float>(1) / (lambda.at<float>(0) + lambda.at<float>(1));
			
			R_mat.at<float>(i, j) = R;
			det = *it_IxIx * *it_IyIy - *it_IxIy * *it_IxIy;
			trac = *it_IxIx + *it_IyIy;
			R = det / trac;
			++it_IxIx;
			++it_IxIy;
			++it_IyIy;
			if (R > debug_max)
			{
				cout << "lambda:" << lambda << endl;
				debug_max = R;
			}

			//cout << H_mat.at<float>(0, 0) << endl;
			if (R > thres)
			{
				harris_points.push_back(Point(j, i));
				circle(src_gray, Point(j, i), 5, Scalar(0), 2, 8, 0);
				//cout << "get one harris" << endl;
			}
		}
	}
	cout << "size of element:" << sizeof(IyIy.at<float>(0,0)) << endl;
	cout << "r_max:" << debug_max << endl;
	cout << "nRow:" << nRows << ", nCols:" << nCols << endl;
	//for (i = 0; i < src_gray.size().height; i++)
	//{
	//	for (j = 0; j < src_gray.size().width; j++)
	//	{
	//		H_mat.at<float
	//	}
	//}

	Mat m1(2, 2, CV_32FC1);
	m1.at<float>(0, 0) = 1;
	m1.at<float>(0, 1) = 2;
	m1.at<float>(1, 0) = 3;
	m1.at<float>(1, 1) = 4;

	Mat m2(2, 2, CV_32F);
	m2 = m1;
	cout << m1.mul(m2) << endl;
	imshow("IxIx", IxIx);
	imshow("IyIy", IyIy);
	imshow("IxIy", IxIy);
	imshow("R_mat", R_mat);
	imshow("harris corner", src_gray);
	int c = waitKey(3000);
}