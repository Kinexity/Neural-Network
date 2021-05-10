#include <fstream>
#include <array>
#include <iostream>
#include <filesystem>
#include <execution>
#include <algorithm>
#include <random>
#include <conio.h>
#include "C_Time_Counter.h"
#include "neural_network.h"

void print_img(const std::vector<double>& img) {
	for (int i = 0; i < 28; i++) {
		for (int j = 0; j < 28; j++) {
			auto px = img[i * 28 + j];
			std::cout << (px > 0.5 ? "#" : (px > 0 ? "*" : " "));
		}
		std::cout << '\n';
	}
}

int main() {
	std::array<std::wstring, 4>
		filenames = { L"train-images.idx3-ubyte",L"train-labels.idx1-ubyte",L"t10k-images.idx3-ubyte",L"t10k-labels.idx1-ubyte" };
	std::ifstream
		images_file,
		labels_file;
	std::vector<std::pair<std::vector<double>, uint8_t>>
		train_data,
		test_data;
	const size_t img_size = 784;
	std::vector<std::byte>
		temp_raw_data(img_size);
	std::vector<double>
		temp_data(img_size);
	uint8_t
		temp_label;
	std::filesystem::path nn_data_path = L"C:\\Users\\26kuba05\\source\\repos\\SpacePhysicsSimulator\\nn_data";
	images_file.open(nn_data_path / filenames[0], std::ios::binary);
	labels_file.open(nn_data_path / filenames[1], std::ios::binary);
	images_file.seekg(16, std::ios::beg);
	labels_file.seekg(8, std::ios::beg);
	do {
		images_file.read((char*)temp_raw_data.data(), img_size);
		char read_buf;
		labels_file.read((char*)&temp_label, 1);
		std::transform(std::execution::par_unseq, temp_raw_data.begin(), temp_raw_data.end(), temp_data.begin(), [&](std::byte& pixel) {
			return double(pixel) / 255;
		});
		train_data.push_back({ temp_data, temp_label });
	} while (!images_file.eof());
	images_file.close();
	labels_file.close();
	images_file.open(nn_data_path / filenames[2], std::ios::binary);
	labels_file.open(nn_data_path / filenames[3], std::ios::binary);
	images_file.seekg(16, std::ios::beg);
	labels_file.seekg(8, std::ios::beg);
	do {
		images_file.read((char*)temp_raw_data.data(), img_size);
		char read_buf;
		labels_file.read((char*)&temp_label, 1);
		std::transform(std::execution::par_unseq, temp_raw_data.begin(), temp_raw_data.end(), temp_data.begin(), [&](std::byte& pixel) {
			return double(pixel) / 255;
		});
		test_data.push_back({ temp_data, temp_label });
	} while (!images_file.eof());
	images_file.close();
	labels_file.close();
	std::random_device rnd;
	std::mt19937_64 mt{ rnd() };
	if (false) {
		std::cout << "Train sample\n";
		std::vector<std::pair<std::vector<double>, uint8_t>> smp;
		std::sample(train_data.begin(), train_data.end(), std::back_inserter(smp), 10, mt);
		for (auto& [img, lb] : smp) {
			print_img(img);
			std::cout << (int)lb << '\n';
		}
		std::cout << "Test sample\n";
		std::sample(test_data.begin(), test_data.end(), std::back_inserter(smp), 10, mt);
		for (auto& [img, lb] : smp) {
			print_img(img);
			std::cout << (int)lb << '\n';
		}
	}
	std::cout << "Data loaded\n";
	for (int i = 0; i < 10; i++) {
		std::cout << "Attempt #" << i << '\n'; 
		neural_network nn({ img_size,50,30,10 });
		double cost = 0.;
		//std::cout << "Starting testing\n";
		//tc.start();
		//for (size_t i = 0; i < test_data.size(); i++) {
		//	auto& [data, label] = test_data[i];
		//	auto&& res = nn.calc_result(data);
		//	res[(int)label] -= 1;
		//	auto l_cost = std::transform_reduce(res.begin(), res.end(), 0., std::plus<>(), [&](double x) {return x * x; });
		//	cost += l_cost;
		//}
		//cost /= test_data.size();
		//tc.stop();
		//std::cout << "Average cost: " << cost << '\n';
		//std::cout << "Testing time " << tc.measured_timespan().count() << " s\n";
		std::cout << "Starting training\n";
		tc.start();
		for (auto& [data, label] : train_data) {
			std::vector<double> vec(10);
			vec[label] = 1;
			nn.train(data, vec);
		}
		tc.stop();
		std::cout << "Training time " << tc.measured_timespan().count() << " s\n";
		std::cout << "Retesting\n";
		tc.start();
		cost = 0.;
		size_t accurate_guesses = 0;
		for (auto& [data, label] : test_data) {
			auto&& res = nn.calc_result(data);
			//std::cout << "Number: " << (int)label << "	Guess:" << std::distance(res.begin(), std::max_element(res.begin(), res.end())) << '\n';
			auto max_it = std::max_element(res.begin(), res.end());
			accurate_guesses += (label == std::distance(res.begin(), max_it) /*&& *max_it > 0.9*/);
			res[(int)label] -= 1;
			auto l_cost = std::transform_reduce(res.begin(), res.end(), 0., std::plus<>(), [&](double x) {return x * x; });
			cost += l_cost;
		}
		cost /= test_data.size();
		tc.stop();
		std::cout << "Average cost: " << cost << '\n';
		std::cout << "Accurancy: " << (double)accurate_guesses / test_data.size() << '\n';
		std::cout << "Testing time " << tc.measured_timespan().count() << " s\n\n\n\n";
	}
}