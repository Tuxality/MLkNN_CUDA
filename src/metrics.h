/*
 *  MLkNN Metrics
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

#ifndef __METRICS_H
#define __METRICS_H

namespace Metrics {

typedef struct {
    std::string method;

    struct {
    	value_t time;
    } global;

    struct {
    	value_t subset_accuracy;
    	value_t hamming_loss;
    } instance;
} results_t;

typedef void (*performance_worker_t)(results_t* results, Dataset& test, int* predict);
#define METRICS_IMP(type, name)             calculate_##type_##name
#define METRICS_IMP_(type, name)            void METRICS_IMP(type,name)(results_t* results, Dataset& test, int* predict)
#define METRICS_STORE(type, name, value)    results->type.name += value

METRICS_IMP_(instance, subset_accuracy);
METRICS_IMP_(instance, hamming_loss);

bool prepare(results_t* results, std::string* method_names, unsigned int method_count);
bool populate(results_t* results, Dataset& test, int* predict, Timer& t);
bool calculate(results_t* results, unsigned int method_count, int kCV);
void print(results_t* results, unsigned int method_count);

};

#endif // __METRICS_H
