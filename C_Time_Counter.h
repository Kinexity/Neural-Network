#pragma once
#ifndef C_T_C
#define C_T_C
#include <chrono>
#include <iostream>

namespace PCL {
	class C_Time_Counter {
	private:
		std::chrono::high_resolution_clock::time_point
			beg,
			end;
		bool
			counting_started = false;
	public:
		C_Time_Counter() = default;
		~C_Time_Counter() = default;
		inline void
			start(),
			stop();
		inline std::chrono::duration<double, std::ratio<1, 1>>
			measured_timespan() noexcept;
	};

	inline void C_Time_Counter::start() {
		if (counting_started) {
			throw std::runtime_error("Counting already started!");
		}
		else {
			beg = std::chrono::high_resolution_clock::now();
			counting_started = true;
		}
	}

	inline void C_Time_Counter::stop() {
		if (!counting_started) {
			throw std::runtime_error("Counting not started!");
		}
		else {
			end = std::chrono::high_resolution_clock::now();
			counting_started = false;
		}
	}

	inline std::chrono::duration<double, std::ratio<1, 1>> C_Time_Counter::measured_timespan() noexcept {
		return std::chrono::duration<double, std::ratio<1, 1>>(end - beg);
	}
}

inline PCL::C_Time_Counter tc;

#endif // !C_T_C