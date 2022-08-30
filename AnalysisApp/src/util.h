#pragma once

#include <string>
#include <vector>


//template<typename num_t>
//inline static num_t vec_itr_avg(std::vector<num_t>::iterator start, std::vector<num_t>::iterator end) {
//	num_t sum{ 0 };
//	size_t count{ 0 };
//	for (; start < end; start++) {
//		sum += *start;
//		count++;
//	}
//	return sum / count;
//}
////Prereq: low is higher than the lowest expected val, high is lower than the highest expected value
//template<typename num_t>
//inline static void vec_itr_hla(std::vector<num_t>::iterator start, std::vector<num_t>::iterator end, num_t& high, num_t& low, num_t& avg) {
//	avg = 0;
//	size_t count{ 0 };
//	for (; start < end; start++) {
//		avg += *start;
//		if (*start > high) { high = *start; }
//		if (*start < low) { low = *start; }
//		count++;
//	}
//	avg /= count;
//}

bool openFile(std::string& f);
bool saveFile(std::string& f);