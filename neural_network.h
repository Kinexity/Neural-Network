#pragma once
#ifndef neural_network_h
#define neural_network_h
#include <vector>
#include <algorithm>
#include <numeric>
#include <execution>
#include <functional>
#include <numeric>
#include <iostream>
#include <Eigen/Dense>

class neural_network {
private:
	std::vector<size_t>
		layer_sizes;
	std::vector<Eigen::MatrixXd>
		weights,
		weights_err;
	std::vector<Eigen::VectorXd>
		biases,
		biases_err; //is equal to neuron error
	double
		learning_coef = 0.01;
	std::function<double(double)>
		sigmoid = [&](double x) { return std::tanh(x); },
		sigmoid_derivate = [&](double x) { return 1 / std::pow(std::cosh(x), 2); };
	std::pair<Eigen::VectorXd, Eigen::VectorXd>
		calc_next_layer_with_z(Eigen::VectorXd data, size_t layer_index);
	std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>>
		calc_result_with_z(Eigen::VectorXd data);
	Eigen::VectorXd
		calc_result_internal(Eigen::VectorXd data) const;
public:
	neural_network(std::vector<size_t> layer_sizes_arg);
	neural_network(const neural_network&) = default;
	void
		train(const std::vector<double>& data, std::vector<double> expected_result);
	std::vector<double>
		calc_result(std::vector<double> data) const;
};

#endif // !neural_network_h