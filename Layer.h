#pragma once
#include <cmath>
#include <iostream>
#include <functional>
#include <Eigen/Dense>

class Layer {
protected:
	Layer
		* previous = nullptr;
public:
	Layer(Layer* previous_ptr);
	virtual void
		backpropagation(Eigen::VectorXf data) = 0;
	virtual Eigen::VectorXf
		calc_with_save(Eigen::VectorXf& data) = 0;
	virtual Eigen::VectorXf
		calc(Eigen::VectorXf& data) = 0;
};

class Fully_interconnected_layer : public Layer {
private:
	Eigen::MatrixXf
		weights;
	Eigen::VectorXf
		bias,
		bias_err,
		z_buffer,
		a_buffer;
	std::function<float(float)>
		sigmoid = [&](float x) { return 1 / (1 + std::exp(-x)); },
		sigmoid_derivate = [&](float x) {
		auto t = sigmoid(x);
		return t * (1 - t); };
public:
	Fully_interconnected_layer(size_t in_size, size_t out_size, Layer* previous_ptr = nullptr);
	~Fully_interconnected_layer() = default;
	void
		backpropagation(Eigen::VectorXf data);
	Eigen::VectorXf
		calc_with_save(Eigen::VectorXf& data),
		calc(Eigen::VectorXf& data);
};

