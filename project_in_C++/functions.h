#pragma once

#include <iostream>
#include <stdio.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigen>
#include <math.h>
#include <time.h>

inline double calculation_K(Eigen::MatrixXd x, Eigen::MatrixXd y, Eigen::VectorXd w, double h)
{
	// std::cout << "calculation K function" << std::endl;
	// std::cout << x << std::endl;
	// std::cout << y << std::endl;
	// std::cout << (x - y).cwiseProduct(x - y) << std::endl;
	Eigen::MatrixXd res = (-(x - y).cwiseProduct(x - y) * w);
	// std::cout << res.rows() << " " << res.cols() << std::endl;
	// system("pause");
	
	return exp(-((x - y).cwiseProduct(x - y) * w)(0, 0) / h);
}


std::pair<Eigen::MatrixXd, Eigen::VectorXd> WBMS(Eigen::MatrixXd X, double h, double lambda = 1, double tmax_1 = 20, double tmax_2 = 30)
{
	time_t time_start = time(NULL);

	int n = X.rows(), p = X.cols();
	Eigen::VectorXd w = Eigen::VectorXd::Constant(p, 1.0 / p);
	Eigen::VectorXd D = Eigen::VectorXd::Zero(p);
	Eigen::MatrixXd X1 = X, X2 = X;
	for (int t = 0; t < tmax_1; t++)
	{
		for (int i = 0; i < n; i++)
		{
			Eigen::VectorXd K_matrix = Eigen::VectorXd::Zero(n);
			double sum_K = 0;
			if (i % 1000 == 0)
			{
				std::cout << t << " " << i << std::endl;

				time_t time_end = time(NULL);
				double time_diff_sec = difftime(time_end, time_start);
				printf("lib: start time: %s", ctime(&time_start));
				printf("lib: end   time: %s", ctime(&time_end));
				printf("lib: time  diff: %fs.\n", time_diff_sec);
			}


			for (int I = 0; I < n; I++)
			{
				if (I == i)
				{
					//sum_K += 1;
					continue;
				}
				K_matrix[I] = calculation_K(X2.row(i), X2.row(I), w, h);
				// if (I % 1000 == 0)
				//  	std::cout << t << " " << i << " " << I << " calculate K correct" << std::endl;
				// system("pause");
				sum_K += K_matrix[I];
			}
			// std::cout << "0.5 right" << std::endl;
			for (int l = 0; l < p; l++)
			{
				X1(i, l) = (K_matrix.transpose() * X2.col(l))(0, 0);
				X1(i, l) /= sum_K;
			}
		}
		D = (X - X1).cwiseProduct(X - X1).colwise().sum();
		w = (-D / lambda).array().exp();
		w = w / w.sum();
		X2 = X1;
		std::cout << "turn " << t << " done" << std::endl;
		std::cout << D[0] << " " << D[1] << " " << D[2] << std::endl;
 		std::cout << w[0] << " " << w[1] << " " << w[2] << std::endl;
		time_t time_end = time(NULL);
		double time_diff_sec = difftime(time_end, time_start);
		printf("lib: start time: %s", ctime(&time_start));
		printf("lib: end   time: %s", ctime(&time_end));
		printf("lib: time  diff: %fs.\n", time_diff_sec);
	}

	X1 = X;
	X2 = X;

	for (int t = 0; t < tmax_2; t++)
	{
		for (int i = 0; i < n; i++)
		{
			Eigen::VectorXd K_matrix = Eigen::VectorXd::Zero(n);
			double sum_K = 0;
			for (int I = 0; I < n; I++)
			{
				if (I == i) continue;
				K_matrix[I] = calculation_K(X2.row(i), X2.row(I), w, h);
				sum_K += K_matrix[I];
			}
			for (int l = 0; l < p; l++)
			{
				X1(i, l) = (K_matrix.transpose() * X2.col(l))(0, 0);
				X1(i, l) /= sum_K;
			}
		}
		D = (X - X1).cwiseProduct(X - X1).colwise().sum();
		w = (-D / lambda).array().exp();
		w = w / w.sum();
		X2 = X1;
		std::cout << "turn " << t << " done" << std::endl;
	}
	return std::pair<Eigen::MatrixXd, Eigen::VectorXd> (X2, w);
}


inline double vector_cdot(std::vector<double> &x, std::vector<double> &y)
{
	double sum = 0;
	int len = x.size();

	for (int i = 0; i < len; i++)
	{
		std::cout << x[i] << std::endl;
		std::cout << y[i] << std::endl;
		sum += x[i] * y[i];
	}

	return sum;
}
inline double calculation_K(std::vector<double> &x, std::vector<double> &y, std::vector<double> &w, double h)
{
	double sum = 0;
	int len = x.size();
	for (int i = 0; i < len; i++)
		sum += (x[i] - y[i]) * (x[i] - y[i]) * w[i];

	return exp(-sum / h);
}

std::pair<std::vector<std::vector<double>>, std::vector<double>> WBMS(std::vector<std::vector<double>> &X, double h, double lambda = 1, double tmax_1 = 20, double tmax_2 = 30)
{
	time_t time_start = time(NULL);

	int n = X.size(), p = X[0].size();
	std::vector<double> w(p, 1.0 / p);
	std::vector<double> D(p, 0);
	std::vector<std::vector<double>> X1 = X;
	std::vector<std::vector<double>> X2 = X;

	for (int t = 0; t < tmax_1; t++)
	{
		for (int i = 0; i < n; i++)
		{
			std::vector<double> K_matrix(n, 0);
			double sum_K = 0;
			if (i % 1000 == 0)
			{
				std::cout << t << " " << i << std::endl;

				time_t time_end = time(NULL);
				double time_diff_sec = difftime(time_end, time_start);
				printf("lib: start time: %s", ctime(&time_start));
				printf("lib: end   time: %s", ctime(&time_end));
				printf("lib: time  diff: %fs.\n", time_diff_sec);
			}


			for (int I = 0; I < n; I++)
			{
				if (I == i)
				{
					//sum_K += 1;
					continue;
				}
				K_matrix[I] = calculation_K(X2[i], X2[I], w, h);
				// if (I % 1000 == 0)
				//  	std::cout << t << " " << i << " " << I << " calculate K correct" << std::endl;
				// system("pause");
				sum_K += K_matrix[I];
			}

			// for (int I = 0; I < 5; I++)
			// 	std::cout << I << " : " << K_matrix[I] << std::endl;
			// return std::pair<std::vector<std::vector<double>>, std::vector<double>>(X2, w);

			// std::cout << "0.5 right" << std::endl;
			for (int l = 0; l < p; l++)
			{
				X1[i][l] = 0;
				for (int j = 0; j < n; j++)
					X1[i][l] += K_matrix[j] * X2[j][l];
				X1[i][l] /= sum_K;
			}
		}

		for (int l = 0; l < p; l++)
			for (int i = 0; i < n; i++)
				D[l] += (X[i][l] - X1[i][l]) * (X[i][l] - X1[i][l]);
		// std::cout << D[0] << " " << D[1] << " " << D[2] << std::endl;
		// system("pause");
		double w_sum = 0;
		for (int l = 0; l < p; l++)
		{
			w[l] = exp(-D[l] / lambda);
			w_sum += w[l];
		}
		for (int l = 0; l < p; l++)
			w[l] = w[l] / w_sum;
		X2 = X1;
		std::cout << "turn " << t << " done" << std::endl;
	}

	X1 = X;
	X2 = X;

	for (int t = 0; t < tmax_2; t++)
	{
		for (int i = 0; i < n; i++)
		{
			std::vector<double> K_matrix(n);
			double sum_K = 0;
			if (i % 1000 == 0)
			{
				std::cout << t << " " << i << std::endl;

				time_t time_end = time(NULL);
				double time_diff_sec = difftime(time_end, time_start);
				printf("lib: start time: %s", ctime(&time_start));
				printf("lib: end   time: %s", ctime(&time_end));
				printf("lib: time  diff: %fs.\n", time_diff_sec);
			}


			for (int I = 0; I < n; I++)
			{
				if (I == i) continue;
				K_matrix[I] = calculation_K(X2[i], X2[I], w, h);
				// if (I % 1000 == 0)
				//  	std::cout << t << " " << i << " " << I << " calculate K correct" << std::endl;
				// system("pause");
				sum_K += K_matrix[I];
			}
			// std::cout << "0.5 right" << std::endl;
			for (int l = 0; l < p; l++)
			{
				X1[i][l] = 0;
				for (int j = 0; j < n; j++)
					X1[i][l] += K_matrix[j] * X2[j][l];
				X1[i][l] /= sum_K;
			}
		}

		for (int l = 0; l < p; l++)
			for (int i = 0; i < n; i++)
				D[l] += (X[i][l] - X1[i][l]) * (X[i][l] - X1[i][l]);

		double w_sum = 0;
		for (int l = 0; l < p; l++)
		{
			w[l] = exp(-D[l] / lambda);
			w_sum += w[l];
		}
		for (int l = 0; l < p; l++)
			w[l] = w[l] / w_sum;
		X2 = X1;
		std::cout << "turn " << t << " done" << std::endl;
	}

	return std::pair<std::vector<std::vector<double>>, std::vector<double>>(X2, w);
}

std::pair<std::vector<std::vector<double>>, std::vector<double>> WBMS_multithread(std::vector<std::vector<double>> &X, double h, double lambda = 1, double tmax_1 = 20, double tmax_2 = 30)
{
	time_t time_start = time(NULL);

	int n = X.size(), p = X[0].size();
	std::vector<double> w(p, 1.0 / p);
	std::vector<double> D(p, 0);
	std::vector<std::vector<double>> X1 = X;
	std::vector<std::vector<double>> X2 = X;

	for (int t = 0; t < tmax_1; t++)
	{
		for (int i = 0; i < n; i++)
		{
			std::vector<double> K_matrix(n);
			double sum_K = 0;
			if (i % 1000 == 0)
			{
				std::cout << t << " " << i << std::endl;

				time_t time_end = time(NULL);
				double time_diff_sec = difftime(time_end, time_start);
				printf("lib: start time: %s", ctime(&time_start));
				printf("lib: end   time: %s", ctime(&time_end));
				printf("lib: time  diff: %fs.\n", time_diff_sec);
			}

			auto K_matrix_MultiThread = [&](int min_I, int max_I) {
				for (int I = min_I; I < max_I; I++)
				{
					if (I == i) continue;
					int len = X2[i].size();
					double sum = 0;
					for (int idx = 0; idx < len; idx++)
						sum += (X2[i][idx] - X2[I][idx]) * (X2[i][idx] - X2[I][idx]) * w[idx];
					K_matrix[I] = exp(-sum / h);
					// if (I % 1000 == 0)
					//  	std::cout << t << " " << i << " " << I << " calculate K correct" << std::endl;
					// system("pause");
					sum_K += K_matrix[I];
				}
			};
			const int block = 4;
			std::thread th[block];
			int stride = n / block + 1;

			// std::cout << block << std::endl;
			for (int core_idx = 0; core_idx < block; core_idx++)
			{
				th[core_idx] = std::thread(K_matrix_MultiThread, core_idx * stride, std::min(core_idx * stride + stride, n));
				time_t time_temp = time(NULL);
				//printf("lib: temp  time: %s", ctime(&time_temp));
			}

			for (int core_idx = 0; core_idx < block; core_idx++)
				th[core_idx].join();
			// std::cout << "0.5 right" << std::endl;
			for (int l = 0; l < p; l++)
			{
				X1[i][l] = 0;
				for (int j = 0; j < n; j++)
					X1[i][l] += K_matrix[j] * X2[j][l];
				X1[i][l] /= sum_K;
			}
		}

		for (int l = 0; l < p; l++)
			for (int i = 0; i < n; i++)
				D[l] += (X[i][l] - X1[i][l]) * (X[i][l] - X1[i][l]);

		double w_sum = 0;
		for (int l = 0; l < p; l++)
		{
			w[l] = exp(-D[l] / lambda);
			w_sum += w[l];
		}
		for (int l = 0; l < p; l++)
			w[l] = w[l] / w_sum;
		X2 = X1;
		std::cout << "turn " << t << " done" << std::endl;
	}

	X1 = X;
	X2 = X;

	for (int t = 0; t < tmax_2; t++)
	{
		for (int i = 0; i < n; i++)
		{
			std::vector<double> K_matrix(n);
			double sum_K = 0;
			if (i % 1000 == 0)
			{
				std::cout << t << " " << i << std::endl;

				time_t time_end = time(NULL);
				double time_diff_sec = difftime(time_end, time_start);
				printf("lib: start time: %s", ctime(&time_start));
				printf("lib: end   time: %s", ctime(&time_end));
				printf("lib: time  diff: %fs.\n", time_diff_sec);
			}


			for (int I = 0; I < n; I++)
			{
				if (I == i) continue;
				K_matrix[I] = calculation_K(X2[i], X2[I], w, h);
				// if (I % 1000 == 0)
				//  	std::cout << t << " " << i << " " << I << " calculate K correct" << std::endl;
				// system("pause");
				sum_K += K_matrix[I];
			}
			// std::cout << "0.5 right" << std::endl;
			for (int l = 0; l < p; l++)
			{
				X1[i][l] = 0;
				for (int j = 0; j < n; j++)
					X1[i][l] += K_matrix[j] * X2[j][l];
				X1[i][l] /= sum_K;
			}
		}

		for (int l = 0; l < p; l++)
			for (int i = 0; i < n; i++)
				D[l] += (X[i][l] - X1[i][l]) * (X[i][l] - X1[i][l]);

		double w_sum = 0;
		for (int l = 0; l < p; l++)
		{
			w[l] = exp(-D[l] / lambda);
			w_sum += w[l];
		}
		for (int l = 0; l < p; l++)
			w[l] = w[l] / w_sum;
		X2 = X1;
		std::cout << "turn " << t << " done" << std::endl;
		std::cout << w[0] << " " << w[1] << " " << w[2] << std::endl;
		time_t time_end = time(NULL);
		double time_diff_sec = difftime(time_end, time_start);
		printf("lib: start time: %s", ctime(&time_start));
		printf("lib: end   time: %s", ctime(&time_end));
		printf("lib: time  diff: %fs.\n", time_diff_sec);
	}

	return std::pair<std::vector<std::vector<double>>, std::vector<double>>(X2, w);
}


std::pair<std::vector<std::vector<double>>, std::vector<double>> WBMS_multithread_2(std::vector<std::vector<double>> &X, double h, double lambda = 1, double tmax_1 = 20, double tmax_2 = 30)
{
	int block_number = 8;
	time_t time_start = time(NULL);

	int n = X.size(), p = X[0].size();
	std::vector<double> w(p, 1.0 / p);
	std::vector<double> D(p, 0);
	std::vector<std::vector<double>> X1 = X;
	std::vector<std::vector<double>> X2 = X;
	std::vector<double> D_copy(p, 0);

	for (int t = 0; t < tmax_1; t++)
	{
		auto K_matrix_MultiThread = [&](int min_i, int max_i) {
			for (int i = min_i; i < max_i; i++)
			{
				std::vector<double> K_matrix(n, 0);
				double sum_K = 0;
				if (i % 1000 == 0 && false)
				{
					std::cout << t << " " << i << std::endl;

					time_t time_end = time(NULL);
					double time_diff_sec = difftime(time_end, time_start);
					printf("lib: start time: %s", ctime(&time_start));
					printf("lib: end   time: %s", ctime(&time_end));
					printf("lib: time  diff: %fs.\n", time_diff_sec);
				}


				for (int I = 0; I < n; I++)
				{
					if (I == i)
					{
						// sum_K += 1;
						continue;
					}
					int len = X2[i].size();
					double sum = 0;
					for (int idx = 0; idx < len; idx++)
					{
						sum += (X2[i][idx] - X2[I][idx]) * (X2[i][idx] - X2[I][idx]) * w[idx];
					}
					K_matrix[I] = exp(-sum / h);
					// K_matrix[I] = calculation_K(X2[i], X2[I], w, h);
					// if (I % 1000 == 0)
					//  	std::cout << t << " " << i << " " << I << " calculate K correct" << std::endl;
					// system("pause");
					sum_K += K_matrix[I];
				}
				// std::cout << "0.5 right" << std::endl;
				for (int l = 0; l < p; l++)
				{
					X1[i][l] = 0;
					for (int j = 0; j < n; j++)
						X1[i][l] += K_matrix[j] * X2[j][l];
					X1[i][l] /= sum_K;
				}
			}
		};
		const int block = 8;
		std::thread th[block];
		int stride = n / block + 1;

		// std::cout << block << std::endl;
		for (int core_idx = 0; core_idx < block; core_idx++)
		{
			th[core_idx] = std::thread(K_matrix_MultiThread, core_idx * stride, std::min(core_idx * stride + stride, n));
			//time_t time_temp = time(NULL);
			//printf("lib: temp  time: %s", ctime(&time_temp));
		}
		for (int core_idx = 0; core_idx < block; core_idx++)
			th[core_idx].join();


		double D_min = DBL_MAX;
		for (int l = 0; l < p; l++)
		{
			D[l] = 0.0;
			for (int i = 0; i < n; i++)
				D[l] += (X[i][l] - X1[i][l]) * (X[i][l] - X1[i][l]);
			D_min = std::min(D_min, D[l]);
		}

		double w_sum = 0;
		for (int l = 0; l < p; l++)
		{
			w[l] = exp(-(D[l] - D_min) / lambda);
			w_sum += w[l];
		}
		for (int l = 0; l < p; l++)
			w[l] = w[l] / w_sum;
		X2 = X1;
		std::cout << "turn " << t << " done" << std::endl;
		std::cout << X2[0][0] << " " << X2[0][1] << " " << X2[0][2] << std::endl;
		std::cout << X2[1][0] << " " << X2[1][1] << " " << X2[1][2] << std::endl;
		time_t time_end = time(NULL);
		double time_diff_sec = difftime(time_end, time_start);
		std::cout << D[0] << " " << D[1] << " " << D[2] << std::endl;
		std::cout << w[0] << " " << w[1] << " " << w[2] << std::endl;
		printf("lib: start time: %s", ctime(&time_start));
		printf("lib: end   time: %s", ctime(&time_end));
		printf("lib: time  diff: %fs.\n", time_diff_sec);
		double diff = std::abs(D[0] - D_copy[0]) + std::abs(D[1] - D_copy[1]) + std::abs(D[2] - D_copy[2]);
		double diff_pre = diff / (D_copy[0] + D_copy[1] + D_copy[2] + 1e-8);
		if (diff_pre < 1e-3)
		{
			std::cout << "Converge in turn " << t << std::endl;
			break;
		}
		D_copy[0] = D[0]; D_copy[1] = D[1]; D_copy[2] = D[2];
	}

	X1 = X;
	X2 = X;

	D[0] = 0.0; D[1] = 0.0; D[2] = 0.0;
	for (int t = 0; t < tmax_2; t++)
	{
		auto K_matrix_MultiThread = [&](int min_i, int max_i) {
			for (int i = min_i; i < max_i; i++)
			{
				std::vector<double> K_matrix(n);
				double sum_K = 0;
				if (i % 1000 == 0 && false)
				{
					std::cout << t << " " << i << std::endl;

					time_t time_end = time(NULL);
					double time_diff_sec = difftime(time_end, time_start);
					printf("lib: start time: %s", ctime(&time_start));
					printf("lib: end   time: %s", ctime(&time_end));
					printf("lib: time  diff: %fs.\n", time_diff_sec);
				}


				for (int I = 0; I < n; I++)
				{
					if (I == i) continue;
					int len = X2[i].size();
					double sum = 0;
					for (int idx = 0; idx < len; idx++)
					{
						sum += (X2[i][idx] - X2[I][idx]) * (X2[i][idx] - X2[I][idx]) * w[idx];
					}
					K_matrix[I] = exp(-sum / h);
					// K_matrix[I] = calculation_K(X2[i], X2[I], w, h);
					// if (I % 1000 == 0)
					//  	std::cout << t << " " << i << " " << I << " calculate K correct" << std::endl;
					// system("pause");
					sum_K += K_matrix[I];
				}
				// std::cout << "0.5 right" << std::endl;
				for (int l = 0; l < p; l++)
				{
					X1[i][l] = 0;
					for (int j = 0; j < n; j++)
						X1[i][l] += K_matrix[j] * X2[j][l];
					X1[i][l] /= sum_K;
				}
			}
		};
		const int block = 8;
		std::thread th[block];
		int stride = n / block + 1;

		// std::cout << block << std::endl;
		for (int core_idx = 0; core_idx < block; core_idx++)
		{
			th[core_idx] = std::thread(K_matrix_MultiThread, core_idx * stride, std::min(core_idx * stride + stride, n));
			//time_t time_temp = time(NULL);
			//printf("lib: temp  time: %s", ctime(&time_temp));
		}
		for (int core_idx = 0; core_idx < block; core_idx++)
			th[core_idx].join();

		double D_min = DBL_MAX;
		for (int l = 0; l < p; l++)
		{
			D[l] = 0.0;
			for (int i = 0; i < n; i++)
				D[l] += (X[i][l] - X1[i][l]) * (X[i][l] - X1[i][l]);
			D_min = std::min(D_min, D[l]);
		}

		double w_sum = 0;
		for (int l = 0; l < p; l++)
		{
			w[l] = exp(-(D[l] - D_min) / lambda);
			w_sum += w[l];
		}
		for (int l = 0; l < p; l++)
			w[l] = w[l] / w_sum;
		X2 = X1;
		std::cout << "turn " << t << " done" << std::endl;
		std::cout << w[0] << " " << w[1] << " " << w[2] << std::endl;
		time_t time_end = time(NULL);
		double time_diff_sec = difftime(time_end, time_start);
		printf("lib: start time: %s", ctime(&time_start));
		printf("lib: end   time: %s", ctime(&time_end));
		printf("lib: time  diff: %fs.\n", time_diff_sec);
		double diff = std::abs(D[0] - D_copy[0]) + std::abs(D[1] - D_copy[1]) + std::abs(D[2] - D_copy[2]);
		double diff_pre = diff / (D_copy[0] + D_copy[1] + D_copy[2] + 1e-8);
		if (diff_pre < 1e-3)
		{
			std::cout << "Converge in turn " << t << std::endl;
			break;
		}
		D_copy[0] = D[0]; D_copy[1] = D[1]; D_copy[2] = D[2];
	}

	return std::pair<std::vector<std::vector<double>>, std::vector<double>>(X2, w);
}