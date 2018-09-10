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

#include "knn.h"

namespace Metrics {

#define METRICS_PREPARE(type, name)     		results[i].type.name = 0.0f;

#define METRICS_NORMALIZE(type, name)   		results[i].type.name /= kCV;

#define METRICS_PRINT(str, type, name, m, f)	LOGA("%s:\n", str);                                                             			\
                                                                                                                        					\
                                                for(unsigned int i = 0; i<method_count; i++) {                                  			\
                                                	LOGA("%.10s: %.2f %s\n", results[i].method.c_str(), results[i].type.name * m, f);  		\
                                                }                                                                               			\
                                                                                                                        					\
                                                LOGA("\n");

static performance_worker_t metrics_fn[] = {
    METRICS_IMP(instance, subset_accuracy),
    METRICS_IMP(instance, hamming_loss)
};

bool prepare(results_t* results, std::string* method_names, unsigned int method_count) {
    for(unsigned int i = 0; i<method_count; i++) {
        results[i].method = method_names[i];

        METRICS_PREPARE(global, time)
        METRICS_PREPARE(instance, subset_accuracy)
        METRICS_PREPARE(instance, hamming_loss)
    }

    return true;
}

bool populate(results_t* results, Dataset& test, int* predict, Timer& t) {
	value_t time = (value_t)((value_t)t.getDiffTime() / (value_t)1e6);
	results->global.time += (value_t)time;
	LOGA("Time:            %f\n", time);

    for(auto metric: metrics_fn) {
        metric(results, test, predict);
    }

    return true;
}

bool calculate(results_t* results, unsigned int method_count, int kCV) {
    if(kCV <= 0) {
        return false;
    }

    for(unsigned int i = 0; i<method_count; i++) {
    	METRICS_NORMALIZE(global, time)
        METRICS_NORMALIZE(instance, subset_accuracy)
        METRICS_NORMALIZE(instance, hamming_loss)
    }

    return true;
}

void print(results_t* results, unsigned int method_count) {
	METRICS_PRINT("Time", global, time, 1, "ms")
    METRICS_PRINT("Subset accuracy", instance, subset_accuracy, 100, "");
    METRICS_PRINT("Hamming loss", instance, hamming_loss, 100, "");
}

};
