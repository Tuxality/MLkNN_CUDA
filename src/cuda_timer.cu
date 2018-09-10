/*
 *  MLkNN CUDA Timer
 *
 *  Author:
 *   Przemyslaw Skryjomski <skryjomskipl@vcu.edu>
 *
 *   This program is free software; you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation; either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program; if not, see <https://www.gnu.org/licenses/>.
 *
 */

#include "cuda_timer.h"

#include <cstdio>

cudaTimer::cudaTimer(std::string str): m_str(str) {
	cudaEventCreate(&m_start);
	cudaEventCreate(&m_end);
	m_ms = 0;
}

cudaTimer::~cudaTimer() {
	cudaEventDestroy(m_start);
	cudaEventDestroy(m_end);
}

void cudaTimer::start() {
	cudaEventRecord(m_start);
}

void cudaTimer::stop() {
	cudaEventRecord(m_end);
	cudaEventSynchronize(m_end);
	cudaEventElapsedTime(&m_ms, m_start, m_end);
}

float cudaTimer::get() {
	return m_ms;
}

void cudaTimer::print() {
	printf("%s Elapsed Time: %.2f ms\n", m_str.c_str(), m_ms);
}

void cudaTimer::print_ratio(cudaTimer& t) {
	float ratio = 0;

	if(m_ms != 0.0f) {
		ratio = t.get() / this->get();
	}

	printf("Ratio:            %.2fx\n", ratio);
}
