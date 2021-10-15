#include <fstream>
#include <array>
#include <iostream>
#include <filesystem>
#include <execution>
#include <algorithm>
#include <random>
#include <conio.h>
#include <sstream>
#include <set>
#include <json/json.h>
#include "C_Time_Counter.h"
#include "feed_forward_neural_network.h"
#include "sieve.h"
#include "C_Random.h"
#include <bit>
#include <bitset>
#include <limits>

void print_img(const std::vector<float>& img) {
	for (int i = 0; i < 28; i++) {
		for (int j = 0; j < 28; j++) {
			auto px = img[i * 28 + j];
			std::cout << (px > 0.5 ? "#" : (px > 0 ? "*" : " "));
		}
		std::cout << '\n';
	}
}

void mnist_test() {
	std::vector<std::array<std::wstring, 4>>
		filenames = {
			{
				L"train-images.idx3-ubyte",
				L"train-labels.idx1-ubyte",
				L"t10k-images.idx3-ubyte",
				L"t10k-labels.idx1-ubyte" },
			{
				L"train-images.idx3-ubyte",
				L"train-labels.idx1-ubyte",
				L"t10k-images.idx3-ubyte",
				L"t10k-labels.idx1-ubyte" },
			{
				L"emnist-byclass-train-images-idx3-ubyte",
				L"emnist-byclass-train-labels-idx1-ubyte",
				L"emnist-byclass-test-images-idx3-ubyte",
				L"emnist-byclass-test-labels-idx1-ubyte" },
			{
				L"emnist-letters-train-images-idx3-ubyte",
				L"emnist-letters-train-labels-idx1-ubyte",
				L"emnist-letters-test-images-idx3-ubyte",
				L"emnist-letters-test-labels-idx1-ubyte" } };
	const size_t img_size = 784;
	std::vector<std::pair<std::vector<float>, uint8_t>>
		train_data,
		test_data;
	size_t
		database = 0,
		output_params = 0,
		pred_size = 0,
		epochs = 0,
		batch_size = 0;
	std::fstream("C:\\Users\\26kub\\source\\repos\\Neural Network\\settings.txt") >> database >> output_params >> pred_size >> epochs >> batch_size;
	bool
		batched = false;
	std::vector<std::filesystem::path> nn_data_path = {
		L"C:\\Users\\26kub\\source\\repos\\SpacePhysicsSimulator\\nn_data",
		L"C:\\Users\\26kub\\source\\repos\\fashion-mnist\\data\\fashion",
		L"C:\\Users\\26kub\\source\\repos\\Neural Network\\data" ,
		L"C:\\Users\\26kub\\source\\repos\\Neural Network\\data" };
	{
		std::ifstream
			images_file,
			labels_file;
		std::vector<std::byte>
			temp_raw_data(img_size);
		uint8_t
			temp_label;
		std::map<decltype(temp_label), size_t> labels_map;
		{ // read test data
			images_file.open(nn_data_path[database] / filenames[database][2], std::ios::binary);
			labels_file.open(nn_data_path[database] / filenames[database][3], std::ios::binary);
			images_file.seekg(16, std::ios::beg);
			labels_file.seekg(8, std::ios::beg);
			auto images_test_size = (std::filesystem::file_size(nn_data_path[database] / filenames[database][2]) - 16) / img_size;
			auto labels_test_size = (std::filesystem::file_size(nn_data_path[database] / filenames[database][3]) - 8) / sizeof(temp_label);
			std::vector<decltype(temp_label)> labels(labels_test_size);
			labels_file.read((char*)labels.data(), labels_test_size);
			auto labels_copy = labels;
			std::sort(std::execution::par_unseq, labels_copy.begin(), labels_copy.end());
			auto it = std::unique(std::execution::par_unseq, labels_copy.begin(), labels_copy.end());
			labels_copy.erase(it, labels_copy.end());
			for (int i = 0; i < labels_copy.size(); i++) {
				labels_map.insert({ labels_copy[i],i });
			}
			temp_raw_data.resize(img_size * images_test_size);
			test_data.resize(images_test_size);
			images_file.read((char*)temp_raw_data.data(), img_size * images_test_size);
			std::for_each(std::execution::par_unseq, labels.begin(), labels.end(), [&](uint8_t& label) {
				std::vector<float> temp(img_size);
				auto init_ind = std::distance(labels.data(), &label) * img_size;
				std::transform(temp_raw_data.begin() + init_ind, temp_raw_data.begin() + init_ind + img_size, temp.begin(), [&](std::byte pixel) {
					return double(pixel) / 255;
					});
				test_data[std::distance(labels.data(), &label)] = { temp, labels_map[label] };
				});
			images_file.close();
			labels_file.close();
		}
		{ // read train data
			images_file.open(nn_data_path[database] / filenames[database][0], std::ios::binary);
			labels_file.open(nn_data_path[database] / filenames[database][1], std::ios::binary);
			images_file.seekg(16, std::ios::beg);
			labels_file.seekg(8, std::ios::beg);
			auto images_train_size = (std::filesystem::file_size(nn_data_path[database] / filenames[database][0]) - 16) / img_size;
			auto labels_train_size = (std::filesystem::file_size(nn_data_path[database] / filenames[database][1]) - 8) / sizeof(temp_label);
			std::vector<decltype(temp_label)> labels(labels_train_size);
			labels_file.read((char*)labels.data(), labels_train_size);
			temp_raw_data.resize(img_size * images_train_size);
			train_data.resize(images_train_size);
			images_file.read((char*)temp_raw_data.data(), img_size * images_train_size);
			std::for_each(std::execution::par_unseq, labels.begin(), labels.end(), [&](uint8_t& label)->std::pair<std::vector<float>, uint8_t>&& {
				std::vector<float> temp(img_size);
				auto init_ind = std::distance(labels.data(), &label) * img_size;
				std::transform(temp_raw_data.begin() + init_ind, temp_raw_data.begin() + init_ind + img_size, temp.begin(), [&](std::byte pixel) {
					return double(pixel) / 255;
					});
				train_data[std::distance(labels.data(), &label)] = { temp, labels_map[label] };
				});
			images_file.close();
			labels_file.close();
		}
		output_params = labels_map.size();
	}
	std::cout << "Data loaded\n";
	auto& out = std::cout;
	feed_forward_neural_network nn({ img_size,150,output_params });
	double 
		previous_accurancy = 0.,
		min_accurancy_increase = 0.001;
	uint32_t
		allowed_accurancy_condition_fails = 1,
		accurancy_condition_fails_counter = 0;
	for (int i = 0; i < epochs; i++) {
		out << "Attempt #" << i << '\n';
		double cost = 0.;
		out << "Starting training\n";
		tc.start();
		for (auto& [data, label] : train_data) {
			std::vector<float> vec;
			vec.resize(output_params);
			vec[label] = 1;
			nn.train(data, vec);
		}
		tc.stop();
		out << "Training time " << tc.measured_timespan().count() << " s\n";
		out << "Testing\n";
		cost = 0.;
		size_t accurate_guesses = 0;
		size_t accurate_guesses_high_conf = 0;
		for (auto& [data, label] : test_data) {
			auto&& res = nn.calc_result(data);
			auto max_it = std::max_element(res.begin(), res.end());
			accurate_guesses += (label == std::distance(res.begin(), max_it));
			accurate_guesses_high_conf += (label == std::distance(res.begin(), max_it) && *max_it > 0.9);
			res[label] -= 1;
			auto l_cost = std::transform_reduce(res.begin(), res.end(), 0., std::plus<>(), [&](float x) {return x * x; });
			cost += l_cost;
		}
		cost /= test_data.size();
		out << "Average cost: " << cost << '\n';
		out << "Accurancy (high conf.): " << (double)accurate_guesses_high_conf / test_data.size() << "\n";
		auto accurancy = (double)accurate_guesses / test_data.size();
		out << "Accurancy (overall): " << accurancy << "\n";
		if (accurancy - previous_accurancy < min_accurancy_increase) {
			accurancy_condition_fails_counter++;
			out << "Negligable or no accurancy increase!" << "\n";
		}
		out << "\n\n";
		if (accurancy_condition_fails_counter > allowed_accurancy_condition_fails) {
			out << "Further retraining halted!\n";
			break;
		}
		previous_accurancy = accurancy;
	}
	for (auto& [data, label] : test_data) {
		auto&& res = nn.calc_result(data);
		auto max_it = std::max_element(res.begin(), res.end());
		auto pred = std::distance(res.begin(), max_it);
		if (pred != label && false) {
			print_img(data);
			std::cout << "Expected: " << (int)label << "Prediction: " << pred;
			_getch();
		}
	}
}

void prime_test() {
	size_t N = (size_t)std::numeric_limits<uint32_t>::max() + 1;
	auto primes = PCL::sieve(N);
	feed_forward_neural_network NN({ sizeof(uint32_t),20,20,20,2 });
	size_t
		epochs = 0,
		batch_size = 0,
		data_size = 0,
		test_size = 0,
		null = 0;
	auto uint32_t_to_vec = [&](uint32_t num) {
		std::vector<float> res(8 * sizeof(num));
		std::bitset<32> bs(num);
		for (size_t bit_ind = 0; bit_ind < bs.size(); bit_ind++) {
			res[bit_ind] = bs[bit_ind];
		}
		return res;
	};
	std::fstream("C:\\Users\\26kub\\source\\repos\\Neural Network\\settings.txt") >> null >> null >> null >> epochs >> batch_size >> data_size >> test_size;
	auto& out = std::cout;
	for (size_t epoch = 0; epoch < epochs; epoch++) {
		out << "Attempt #" << epoch << '\n';
		out << "Starting training\n";
		tc.start();
		for (size_t data_ind = 0; data_ind < data_size; data_ind++) {
			uint32_t index = rnd() % N;
			auto state = primes[index];
			std::vector<float> vec(2);
			vec[state] = 1;
			auto data = uint32_t_to_vec(index);
			NN.train(data, vec);
		}
		tc.stop();
		out << "Training time " << tc.measured_timespan().count() << " s\n";
		out << "Testing\n";
		auto cost = 0.;
		size_t accurate_guesses = 0;
		size_t accurate_guesses_high_conf = 0;
		for (size_t test_ind = 0; test_ind < test_size; test_ind++) {
			uint32_t index = rnd() % N;
			auto state = primes[index];
			std::vector<float> vec(2);
			vec[state] = 1;
			auto data = uint32_t_to_vec(index);
			auto&& res = NN.calc_result(data);
			//out << "Number: " << (int)label << "	Guess:" << std::distance(res.begin(), std::max_element(res.begin(), res.end())) << '\n';
			auto max_it = std::max_element(res.begin(), res.end());
			accurate_guesses += (state == std::distance(res.begin(), max_it));
			accurate_guesses_high_conf += (state == std::distance(res.begin(), max_it) && *max_it > 0.9);
			res[state] -= 1;
			auto l_cost = std::transform_reduce(res.begin(), res.end(), 0., std::plus<>(), [&](float x) {return x * x; });
			cost += l_cost;

		}
		cost /= test_size;
		out << "Average cost: " << cost << '\n';
		out << "Accurancy (high conf.): " << (double)accurate_guesses_high_conf / test_size << "\n";
		out << "Accurancy (overall): " << (double)accurate_guesses / test_size << "\n\n\n";
	}
}

int main() {
	mnist_test();
	//prime_test();
}