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

#ifndef __CUDA_TIMER_H
#define __CUDA_TIMER_H

#include <string>

class cudaTimer {
private:
	std::string m_str;
	cudaEvent_t m_start, m_end;
	float m_ms;

public:
	cudaTimer(std::string str);
	~cudaTimer();

	void start();
	void stop();
	float get();

	void print();
	void print_ratio(cudaTimer& t);
};

#endif // __CUDA_TIMER_H
