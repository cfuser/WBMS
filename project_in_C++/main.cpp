#define _USE_MATH_DEFINES
#include "opencv2/imgproc/imgproc_c.h"
#include<mat.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigen>
#include <opencv2/core/eigen.hpp>
#include <opencv.hpp>
#include "functions.h"
#include "utils.h"
#include <mat.h>

bool ReadMatlabMat(double *dst, std::string filePath, std::string matrixName, int width, int height)
{
	MATFile* pmatFile = NULL;
	mxArray* pMxArray = NULL;
	double* matdata;

	pmatFile = matOpen(filePath.c_str(), "r");//打开.mat文件
	if (pmatFile == NULL)
	{
		return false;
	}
	pMxArray = matGetVariable(pmatFile, matrixName.c_str());//获取.mat文件里面名为matrixName的矩阵
	matdata = (double*)mxGetData(pMxArray);//获取指针
	matClose(pmatFile);//close file

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			for (int k = 0; k < 3; k++)
				dst[(i * width + j) * 3 + k] = double(matdata[(i * width + j) * 3 + k]);
		}
	}
	mxDestroyArray(pMxArray);//释放内存
	matdata = NULL;
	return 1;
}

int main()
{
	std::vector<std::string> file_name_vector;

	std::ifstream ifs;
	ifs.open("file.txt", std::ios::in);
	for (int i = 0; i < 105; i++)
	{
		std::string buf;
		getline(ifs, buf);
		buf = buf.substr(0, buf.size() - 12);
		if (i >= 5)
			file_name_vector.push_back(buf);
	}
	// file_name_vector.push_back("0000047");
	// file_name_vector.push_back("0000051");
	// file_name_vector.push_back("0000059");
	// file_name_vector.push_back("0000072");
	// file_name_vector.push_back("0000087");

	double h = 0.1;
	double lambda = 20;
	int tmax_1 = 20;
	int tmax_2 = 30;
	std::cout << "input tmax_1 : ";
	std::cin >> tmax_1;
	std::cout << "input tmax_2 : ";
	std::cin >> tmax_2;
	double epsa = 1e-1;
	std::cout << "input epsa : ";
	std::cin >> epsa;
	std::cout << "input h : ";
	std::cin >> h;
	std::cout << "input lambda : ";
	std::cin >> lambda;

	// std::ofstream write_file;
	// write_file.open("F-measure_result.txt", std::ios::app);

	for (int turn = 0; turn < file_name_vector.size(); turn++)
	{
		std::string file_path = file_name_vector[turn];
		// file_path = "D:/subject/graduate/digital process and model/Project/data/BSDS500/data/images/train/100075.jpg";
		// file_path = "D:/subject/graduate/digital process and model/Project/data/iccv09Data/images/0000047.jpg";
		// bool file_mannual = false;
		// std::cout << "input file_mannual : ";
		// std::cin >> file_mannual;
		// if (file_mannual)
		// {
		// 	std::cout << "input file path : ";
		// 	std::cin >> file_path;
		// }
		std::cout << "-------------" << std::endl;
		std::cout << "file path : " << "data/iccv09Data/images/" + file_path + ".jpg" << std::endl;
		std::cout << std::endl;
		cv::Mat image = cv::imread("data/iccv09Data/images/" + file_path + ".jpg");
		int height = image.rows, width = image.cols;
		//std::vector<cv::Mat> bgr(3);// = { cv::Mat(image.rows, image.cols), cv::Mat(image.rows, image.cols), cv::Mat(image.rows, image.cols) };
		cv::Mat bgr[3];
		cv::split(image, bgr);
		// std::cout << image << std::endl;
		//图像尺寸
		std::cout << "size:" << image.size << std::endl;
		//列宽
		std::cout << "cols:" << image.cols << std::endl;
		//行高
		std::cout << "rows:" << image.rows << std::endl;
		//通道数
		std::cout << "channels:" << image.channels() << std::endl;

		int value_rows = image.rows, value_cols = image.cols, value_features_dim = image.channels();
		//Eigen::Tensor<double, 3, Eigen::RowMajor> pixel(image.rows, image.cols, image.channels());
		std::cout << value_rows << " " << value_cols << std::endl;

		// double** pixel = new double*[value_features_dim];
		// double *pixel_col_mean = new double[value_features_dim];
		// double *pixel_col_std = new double[value_features_dim];

		std::vector<std::vector<double>> pixel(value_rows * value_cols, std::vector<double>(value_features_dim, 0));
		std::vector<double> pixel_col_mean(value_features_dim, 0);
		std::vector<double> pixel_col_std(value_features_dim, 0);

		for (int k = 0; k < value_features_dim; k++)
		{
			// pixel[k] = new double[value_rows * value_cols];
			pixel_col_mean[k] = 0;
			pixel_col_std[k] = 0;

			for (int i = 0; i < value_rows; i++)
				for (int j = 0; j < value_cols; j++)
				{
					double pixel_nor = image.at<cv::Vec3b>(i, j)[k];// / 255.0;
					pixel[i * value_cols + j][k] = pixel_nor;
					pixel_col_mean[k] += pixel_nor / (value_rows * value_cols);
					pixel_col_std[k] += pixel_nor * pixel_nor / (value_rows * value_cols);
				}
			pixel_col_std[k] = sqrt(pixel_col_std[k] - pixel_col_mean[k] * pixel_col_mean[k]);
			for (int i = 0; i < value_rows; i++)
				for (int j = 0; j < value_cols; j++)
				{
					pixel[i * value_cols + j][k] = (pixel[i * value_cols + j][k] - pixel_col_mean[k]) / pixel_col_std[k];
				}
		}

		// std::cout << pixel_col_mean[0] << " " << pixel_col_mean[1] << " " << pixel_col_mean[2] << std::endl;
		// std::cout << pixel_col_std[0] << " " << pixel_col_std[1] << " " << pixel_col_std[2] << std::endl;
		// system("pause");
		// return 0;

		{
			double single_pixel[3] = { pixel[0][0], pixel[0][1], pixel[0][2] };
			single_pixel[0] = (single_pixel[0] * pixel_col_std[0] + pixel_col_mean[0]); // * 255;
			single_pixel[1] = (single_pixel[1] * pixel_col_std[1] + pixel_col_mean[1]); // * 255;
			single_pixel[2] = (single_pixel[2] * pixel_col_std[2] + pixel_col_mean[2]); // * 255;

			std::cout << single_pixel[0] << " " << single_pixel[1] << " " << single_pixel[2] << std::endl;
		}
		std::pair<std::vector<std::vector<double>>, std::vector<double>> res;
		res = WBMS_multithread_2(pixel, h, lambda, tmax_1, tmax_2);
		// res = WBMS(pixel, h, lambda, tmax_1, tmax_2);
		std::vector<std::vector<double>> X_res = res.first;
		std::vector<double> w_res = res.second;

		int pixel_number = value_rows * value_cols;

		std::vector<int> fa(pixel_number);
		for (int i = 0; i < pixel_number; i++)
		{
			fa[i] = i;
			for (int j = 0; j < i; j++)
				if (dist(X_res[i], X_res[j]) < epsa)
				{
					int u = i, v = j;
					Union(fa, i, j);
					break;
				}
		}

		for (int i = 0; i < pixel_number; i++)
			fa[i] = get_Father(fa, i);

		std::set<int> set_Father(fa.begin(), fa.end());
		std::vector<int> class_Father(set_Father.begin(), set_Father.end());
		std::map<int, int> class_reverse;
		int temp_size = class_Father.size();

		for (int i = 0; i < temp_size; i++)
		{
			class_reverse[class_Father[i]] = i;
		}

		for (int i = 0; i < pixel_number; i++)
			fa[i] = class_reverse[fa[i]];

		std::cout << "number of class : " << class_Father.size() << std::endl;

		cv::Mat res_image(value_rows, value_cols, CV_32FC3);

		std::cout << X_res[0][0] << " " << X_res[0][1] << " " << X_res[0][2] << std::endl;
		{
			double single_pixel[3] = { X_res[0][0], X_res[0][1], X_res[0][2] };
			single_pixel[0] = (single_pixel[0] * pixel_col_std[0] + pixel_col_mean[0]); // * 255;
			single_pixel[1] = (single_pixel[1] * pixel_col_std[1] + pixel_col_mean[1]); // * 255;
			single_pixel[2] = (single_pixel[2] * pixel_col_std[2] + pixel_col_mean[2]); // * 255;

			std::cout << single_pixel[0] << " " << single_pixel[1] << " " << single_pixel[2] << std::endl;

			cv::Vec3d color = { single_pixel[0], single_pixel[1], single_pixel[2] };
			std::cout << color << std::endl;
		}
		for (int i = 0; i < value_rows; i++)
			for (int j = 0; j < value_cols; j++)
				for (int k = 0; k < value_features_dim; k++)
				{
					res_image.at<cv::Vec3f>(i, j)[k] = (X_res[i * value_cols + j][k] * pixel_col_std[k] + pixel_col_mean[k]);
					// std::cout << res_image.at<cv::Vec3f>(i, j)[k] << std::endl;
					// system("pause");
				}
		// cv::imwrite("image.jpg", image);
		std::string temp_file_1 = "results/"; temp_file_1 = temp_file_1 + file_path + "_res_image.jpg";
		std::string temp_file_2 = "results/"; temp_file_2 = temp_file_2 + file_path + "_result.mat";
		// std::string temp_file_1 = file_path + "_res_image.jpg";
		// std::string temp_file_2 = file_path + "_result.mat";
		cv::imwrite(temp_file_1.c_str(), res_image);
		MATFile *pmatFile = matOpen(temp_file_2.c_str(), "w");
		mxArray *pMxArray = mxCreateDoubleMatrix(value_rows, value_cols, mxREAL);
		double *pData = (double *)mxCalloc(value_cols * value_rows, sizeof(double));
		for (int i = 0; i < value_rows; i++)
			for (int j = 0; j < value_cols; j++)
			{
				pData[j * value_rows + i] = fa[i * value_cols + j];
			}
		mxSetData(pMxArray, pData);
		matPutVariable(pmatFile, "label", pMxArray);
		matClose(pmatFile);

		std::cout << "class_Father : " << class_Father[0] << " " << class_Father[1] << std::endl;
		std::cout << "class_reverse : " << class_reverse[class_Father[0]] << " " << class_reverse[class_Father[0]] << std::endl;
		std::cout << "fa : " << fa[0] << " " << fa[1] << " " << fa[2] << std::endl;
		for (int i = 0; i < value_rows; i++)
			for (int j = 0; j < value_cols; j++)
			{
				res_image.at<cv::Vec3f>(i, j) /= 255.0;
				// std::cout << res_image.at<cv::Vec3f>(i, j)[k] << std::endl;
				// system("pause");
			}
		// cv::imshow("image", image);
		// cv::imshow("res_image", res_image);
		// cv::waitKey(0);
	}
	system("pause");
	return 0;
}