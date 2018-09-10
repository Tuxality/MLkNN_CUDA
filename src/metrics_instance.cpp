/*
 *  MLkNN Metrics (Instance)
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

#include <cstring>

namespace Metrics {

METRICS_IMP_(instance, subset_accuracy) {
	value_t value = 0;

    for(unsigned int i = 0; i<test.get_num_instances(); i++) {
        unsigned int offset = i * test.get_num_labels();
        int* labels_test = (int*)(test.get_data_Y() + offset);
        int* labels_predict = (int*)(predict + offset);

        if(memcmp(labels_test, labels_predict, sizeof(int) * test.get_num_labels()) == 0)
            value += 1;
    }

    value /= test.get_num_instances();

    LOGA("Subset accuracy: %f\n", value);

    METRICS_STORE(instance, subset_accuracy, value);
}

METRICS_IMP_(instance, hamming_loss) {
	value_t value = 0;

    for(unsigned int i = 0; i<test.get_num_instances(); i++) {
        unsigned int offset = i * test.get_num_labels();
        int* labels_test = (int*)(test.get_data_Y() + offset);
        int* labels_predict = (int*)(predict + offset);

        value_t tmp = 0;

        for(unsigned int j = 0; j<test.get_num_labels(); j++) {
        	if(labels_test[j] != labels_predict[j])
        		tmp += 1;
        }

        tmp /= test.get_num_labels();
        value += tmp;
    }

    value /= test.get_num_instances();

    LOGA("Hamming loss:    %f\n", value);

    METRICS_STORE(instance, hamming_loss, value);
}

};
