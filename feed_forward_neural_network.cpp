#include "feed_forward_neural_network.h"

std::vector<float> conv_vec(Eigen::VectorXf vec) {
	return { vec.data(), vec.data() + vec.rows() };
}

Eigen::VectorXf conv_vec(const std::vector<float> vec) {
	return Eigen::Map<const Eigen::VectorXf, Eigen::Unaligned>(vec.data(), vec.size());
}

std::vector<float> feed_forward_neural_network::calc_result(const std::vector<float> data) const {
	auto data_eugen = conv_vec(data);
	auto res = layers.back()->calc(data_eugen);
	return conv_vec(res);
}

feed_forward_neural_network::feed_forward_neural_network(std::vector<size_t> layer_sizes) {
	layers.reserve(layer_sizes.size() - 1);
	for (int i = 0; i < layer_sizes.size() - 1; i++) {
		layers.push_back(std::make_unique<Fully_interconnected_layer>(layer_sizes[i], layer_sizes[i + 1], (i == 0 ? nullptr : layers[i - 1].get())));
	}
}

void feed_forward_neural_network::train(const std::vector<float>& data, std::vector<float> expected_result) {
	auto expected_result_eigen = conv_vec(expected_result);
	auto data_eigen = conv_vec(data);
	auto res = layers.back()->calc_with_save(data_eigen);
	auto diff = 2 * (expected_result_eigen - res) * learning_coef;
	layers.back()->backpropagation(diff);
}