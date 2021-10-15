#include "feed_forward_neural_network.h"

Eigen::VectorXf softmax(Eigen::VectorXf vec) {
	Eigen::VectorXf vec_exp = vec.unaryExpr([](float x) { return std::exp(x); });
	float sum = vec_exp.sum();
	return vec_exp / sum;
}

Eigen::VectorXf softmax_derivative(Eigen::VectorXf vec) {
	Eigen::VectorXf vec_exp = vec.unaryExpr([](float x) { return std::exp(x); });
	float sum = vec_exp.sum();
	return -vec_exp.unaryExpr([](float x) { return x * x; }) / (sum * sum);
}

std::vector<float> conv_vec(Eigen::VectorXf vec) {
	return { vec.data(), vec.data() + vec.rows() };
}

Eigen::VectorXf conv_vec(const std::vector<float> vec) {
	return Eigen::Map<const Eigen::VectorXf, Eigen::Unaligned>(vec.data(), vec.size());
}

std::vector<float> feed_forward_neural_network::calc_result(const std::vector<float> data) const {
	return conv_vec(calc_result_internal(conv_vec(data)));
}

std::pair<Eigen::VectorXf, Eigen::VectorXf> feed_forward_neural_network::calc_next_layer_with_z(Eigen::VectorXf data, size_t layer_index) {
	auto z_vec = weights[layer_index] * data - biases[layer_index];
	auto a_vec = (!(layer_index == layer_sizes.size() - 2 && is_softmax) ? z_vec.unaryExpr(sigmoid) : softmax(z_vec));
	return { a_vec, z_vec };
}

std::pair<std::vector<Eigen::VectorXf>, std::vector<Eigen::VectorXf>> feed_forward_neural_network::calc_result_with_z(Eigen::VectorXf data) {
	std::pair<std::vector<Eigen::VectorXf>, std::vector<Eigen::VectorXf>> results = { {data},{{}} };
	for (size_t layer_index = 0; layer_index < layer_sizes.size() - 1; layer_index++) {
		auto [a, z] = calc_next_layer_with_z(results.first.back(), layer_index);
		results.first.push_back(a);
		results.second.push_back(z);
	}
	return results;
}

Eigen::VectorXf feed_forward_neural_network::calc_result_internal(Eigen::VectorXf data) const {
	for (size_t layer_index = 0; layer_index < layer_sizes.size() - 1; layer_index++) {
		data = (!(layer_index == layer_sizes.size() - 2 && is_softmax) ?
			(weights[layer_index] * data - biases[layer_index]).unaryExpr(sigmoid) :
			softmax(weights[layer_sizes.size() - 2] * data - biases[layer_sizes.size() - 2]));
	}
	return data;
}

feed_forward_neural_network::feed_forward_neural_network(std::vector<size_t> layer_sizes_arg) : layer_sizes(layer_sizes_arg) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dis(-1.0, 1.0);
	weights.resize(layer_sizes.size() - 1);
	weights_err.resize(layer_sizes.size() - 1);
	biases.resize(layer_sizes.size() - 1);
	biases_err.resize(layer_sizes.size() - 1);
	for (size_t i = 0; i < weights.size(); i++) {
		weights[i] = Eigen::MatrixXf::NullaryExpr(layer_sizes[i + 1], layer_sizes[i], [&]() {return dis(gen); });
		biases[i] = Eigen::VectorXf::Zero(layer_sizes[i + 1]);
	}
}

void feed_forward_neural_network::train(const std::vector<float>& data, std::vector<float> expected_result) {
	auto expected_result_eigen = conv_vec(expected_result);
	auto data_eigen = conv_vec(data);
	auto [a, z] = calc_result_with_z(data_eigen);
	auto diff = 2 * (expected_result_eigen - a.back());
	biases_err[biases_err.size() - 1] = diff.cwiseProduct(!is_softmax ? z.back().unaryExpr(sigmoid_derivate) : softmax_derivative(z.back())) * learning_coef; //learning coef multiplication moved here for perf reasons
	for (size_t layer_index = layer_sizes.size() - 2; layer_index > 0; layer_index--) { //calculate delta/bias error
		biases_err[layer_index - 1] = (weights[layer_index].transpose() * biases_err[layer_index]).cwiseProduct(z[layer_index].unaryExpr(sigmoid_derivate));
	}
	for (size_t layer_index = layer_sizes.size() - 2; layer_index <= layer_sizes.size() - 2; layer_index--) { //calculate weight error
		weights_err[layer_index] = biases_err[layer_index] * a[layer_index].transpose();
	}
	weights += weights_err;
	biases += biases_err;
}

using pmv = std::pair<std::valarray<Eigen::MatrixXf>, std::valarray<Eigen::VectorXf>>;

pmv operator+(pmv p1, pmv p2) {
	return { p1.first + p2.first, p1.second + p2.second };
}