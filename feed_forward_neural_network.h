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
#include <random>
#include <span>
#include <valarray>
#include <Eigen/Dense>

class feed_forward_neural_network {
private:
	std::vector<size_t>
		layer_sizes;
	std::valarray<Eigen::MatrixXf>
		weights,
		weights_err;
	std::valarray<Eigen::VectorXf>
		biases,
		biases_err; //is equal to neuron error
	float
		learning_coef = 0.3;
	std::function<float(float)>
		sigmoid = [&](float x) { return 1 / (1 + std::exp(-x)); },
		sigmoid_derivate = [&](float x) {
		auto t = sigmoid(x);
		return t * (1 - t); };
	bool 
		is_softmax = false;
	std::pair<Eigen::VectorXf, Eigen::VectorXf>
		calc_next_layer_with_z(Eigen::VectorXf data, size_t layer_index);
	std::pair<std::vector<Eigen::VectorXf>, std::vector<Eigen::VectorXf>>
		calc_result_with_z(Eigen::VectorXf data);
	Eigen::VectorXf
		calc_result_internal(Eigen::VectorXf data) const;
public:
	feed_forward_neural_network(std::vector<size_t> layer_sizes_arg);
	feed_forward_neural_network(const feed_forward_neural_network&) = default;
	void
		train(const std::vector<float>& data, std::vector<float> expected_result);
	std::vector<float>
		calc_result(std::vector<float> data) const;
};

#endif // !neural_network_h