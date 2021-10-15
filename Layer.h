#pragma once
#include <cmath>
#include <functional>
#include <Eigen/Dense>

class Layer {
protected:
	Layer
		* previous = nullptr,
		* next = nullptr;
public:
	virtual void
		backpropagation(Eigen::VectorXd data) = 0;
	virtual Eigen::VectorXd
		calc_with_save(Eigen::VectorXd data) = 0,
		calc(Eigen::VectorXd data) = 0;
};

class Fully_interconnected_layer : public Layer {
private:
	Eigen::MatrixXf
		weights,
		weights_err;
	Eigen::VectorXf
		biases,
		biases_err,
		z_buffer,
		a_buffer;
	std::function<double(double)>
		sigmoid,
		sigmoid_deriv;
public:
	Fully_interconnected_layer();
	~Fully_interconnected_layer();
	void
		backpropagation(Eigen::VectorXd data);
	Eigen::VectorXd
		calc_with_save(Eigen::VectorXd data),
		calc(Eigen::VectorXd data);
};

Fully_interconnected_layer::Fully_interconnected_layer() {
}

Fully_interconnected_layer::~Fully_interconnected_layer()
{
}

inline void Fully_interconnected_layer::backpropagation(Eigen::VectorXd data) {
}

inline Eigen::VectorXd Fully_interconnected_layer::calc_with_save(Eigen::VectorXd data)
{
	return Eigen::VectorXd();
}

inline Eigen::VectorXd Fully_interconnected_layer::calc(Eigen::VectorXd data) {
	return weights * (previous == nullptr ? data : previous->calc(data)) + biases;
}

