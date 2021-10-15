#pragma once
#ifndef neural_network_h
#define neural_network_h
#include <vector>
#include <memory>
#include "Layer.h"

class feed_forward_neural_network {
private:
	std::vector<std::unique_ptr<Layer>>
		layers;
	float
		learning_coef = 0.15;
public:
	feed_forward_neural_network(std::vector<size_t> layer_sizes);
	feed_forward_neural_network(const feed_forward_neural_network&) = default;
	void
		train(const std::vector<float>& data, std::vector<float> expected_result);
	std::vector<float>
		calc_result(std::vector<float> data) const;
};

#endif // !neural_network_h