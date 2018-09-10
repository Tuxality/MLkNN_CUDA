/*
 *  MLkNN Timer
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

#include "knn.h"

Timer::Timer() {
    this->reset();
}

void Timer::reset() {
    clock_gettime(CLOCK_MONOTONIC_RAW, &m_start);
}

uint64_t Timer::getDiffTime() {
    clock_gettime(CLOCK_MONOTONIC_RAW, &m_end);

    uint64_t dt = (1000000000L * (m_end.tv_sec - m_start.tv_sec) + m_end.tv_nsec - m_start.tv_nsec);
    return dt;
}
