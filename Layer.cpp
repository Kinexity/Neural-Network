#include "Layer.h"
#include <random>

Fully_interconnected_layer::Fully_interconnected_layer(size_t in_size, size_t out_size, Layer* previous_ptr) : Layer(previous_ptr) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dis(-1.0, 1.0);
	weights = Eigen::MatrixXf::NullaryExpr(out_size, in_size, [&]() {return dis(gen); });
	bias = Eigen::VectorXf::Zero(out_size);
}

inline void Fully_interconnected_layer::backpropagation(Eigen::VectorXf data) {
	bias_err = data.cwiseProduct(z_buffer.unaryExpr(sigmoid_derivate));
	if (previous != nullptr) {
		previous->backpropagation(weights.transpose() * bias_err);
	}
	bias += bias_err;
	weights += bias_err * a_buffer.transpose();
}

inline Eigen::VectorXf Fully_interconnected_layer::calc_with_save(Eigen::VectorXf& data) {
	a_buffer = (previous == nullptr ? data : previous->calc_with_save(data));
	z_buffer = weights * a_buffer - bias;
	return z_buffer.unaryExpr(sigmoid);
}

inline Eigen::VectorXf Fully_interconnected_layer::calc(Eigen::VectorXf& data) {
	return (weights * (previous == nullptr ? data : previous->calc(data)) - bias).unaryExpr(sigmoid);
}

Layer::Layer(Layer* previous_ptr) : previous(previous_ptr) {}
