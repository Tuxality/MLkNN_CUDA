/*
 *  MLkNN CPU Implementation
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

#include <algorithm>
#include <cstring>
#include <cfloat>
#include <cmath>

#define DELETE_IF(x)    if(x)	{			\
							delete[] x;		\
							x = nullptr;	\
						}

#ifdef __DEBUG__
static void dump_id(int i, int id, int k_neighbours, neighbour_t* nn) {
    if(i == id) {
        printf("CPU NN %.3i -> ", id);
    	for(unsigned int k = 0; k<k_neighbours; k++) {
    		printf("%.3i ", nn[k].second);
    	}
    	printf("\n");
    }
}
#endif // __DEBUG__

ClassifierMLkNN_CPU::ClassifierMLkNN_CPU(): ClassifierBase() {
    m_data_X = nullptr;
    m_data_Y = nullptr;

    m_nn = nullptr;

    m_fmax = nullptr;
    m_fmin = nullptr;
    m_ftypes = nullptr;

    m_apriori_P = nullptr;
    m_apriori_NP = nullptr;

    m_aposteriori_P_counts = nullptr;
    m_aposteriori_NP_counts = nullptr;
    m_aposteriori_P = nullptr;
    m_aposteriori_NP = nullptr;
}

ClassifierMLkNN_CPU::~ClassifierMLkNN_CPU() {
    resetModel();
}


bool ClassifierMLkNN_CPU::buildModel(Dataset& train, int parm_k, value_t parm_smooth) {
    resetModel();

    if(!ClassifierBase::buildModel(train, parm_k, parm_smooth))
        return false;

    m_num_instances = train.get_num_instances();
    m_num_attributes = train.get_num_attributes();
    m_num_labels = train.get_num_labels();

    // Allocate memory for dataset
    m_data_X = new value_t[m_num_instances * m_num_attributes];
    m_data_Y = new int[m_num_instances * m_num_labels];

    // Allocate memory for kNN
    m_nn = new neighbour_t[m_parm_k];

    // Allocate memory for min-max scaler
    m_fmax = new value_t[m_num_attributes];
    m_fmin = new value_t[m_num_attributes];
    m_ftypes = new unsigned int[m_num_attributes];

    // Allocate memory for a priori prob
    m_apriori_P = new value_t[m_num_labels];
    m_apriori_NP = new value_t[m_num_labels];

    // Allocate memory for a posteriori prob
    m_aposteriori_P_counts = new int[m_num_labels * (m_parm_k + 1)]();
    m_aposteriori_NP_counts = new int[m_num_labels * (m_parm_k + 1)]();
    m_aposteriori_P = new value_t[m_num_labels * (m_parm_k + 1)];
    m_aposteriori_NP = new value_t[m_num_labels * (m_parm_k + 1)];

    // Copy data
    memcpy(m_data_X, train.get_data_X(), m_num_instances * m_num_attributes * sizeof(value_t));
    memcpy(m_data_Y, train.get_data_Y(), m_num_instances * m_num_labels * sizeof(int));
    memcpy(m_ftypes, train.get_attributes_types(), m_num_attributes * sizeof(unsigned int));

    // Compute values for scaling (used by kNN)
    computeMinMaxValues();

    // Compute apriori probabilities
    computeAprioriProb();

    // Compute aposteriori probabilities
    computeAposterioriProb();

    // Allow syncing model!
    m_model_needsync = true;

    return true;
}

bool ClassifierMLkNN_CPU::syncModel() {
	#ifdef __DEBUG__
    debug();
    #endif // __DEBUG__

	return ClassifierBase::syncModel();
}

int* ClassifierMLkNN_CPU::predict(Dataset& test) {
    if(!m_model_built)
        return nullptr;

    // Get query information
    unsigned int num_instances_test = test.get_num_instances();
    value_t* test_X = test.get_data_X();

    // Allocate memory
    int* predict = new int[num_instances_test * m_num_labels]();

    // Make a prediction!

    for(unsigned int i = 0; i<num_instances_test; i++) {
        // Get neighbours
        unsigned int offset = i * m_num_attributes;

        kNN(m_data_X, &test_X[offset], m_num_instances, m_num_attributes, m_parm_k, m_nn);

        // For each label
        for(unsigned int j = 0; j<m_num_labels; j++) {
            // Get count of neighbours that has this label
            int count = 0;

            for(unsigned k = 0; k<m_parm_k; k++) {
                unsigned int id = m_nn[k].second;
                
                if(m_data_Y[id * m_num_labels + j] == 1)
                    count += 1;
            }

            // Calculate probabilities of the instance belonging to this label or not
            value_t prob_P = m_apriori_P[j] * m_aposteriori_P[j * (m_parm_k + 1) + count];
            value_t prob_NP = m_apriori_NP[j] * m_aposteriori_NP[j * (m_parm_k + 1) + count];

            // if it equals, just assign the label
            unsigned int offset = i * m_num_labels + j;
            
            if(prob_P >= prob_NP) {
                predict[offset] = 1;
            } else {
                predict[offset] = 0;
            }
        }
    }

    if(m_debug_fp) {
    	logDebug("PREDICT\n");
        for(unsigned int i = 0; i<num_instances_test * m_num_labels; i++)
        	logDebug("%i ", predict[i]);
        logDebug("\n\n");
    }

    return predict;
}

bool ClassifierMLkNN_CPU::resetModel() {
    DELETE_IF(m_data_X);
    DELETE_IF(m_data_Y);

    DELETE_IF(m_nn);

    DELETE_IF(m_fmax);
    DELETE_IF(m_fmin);
    DELETE_IF(m_ftypes);

    DELETE_IF(m_apriori_P);
    DELETE_IF(m_apriori_NP);

    DELETE_IF(m_aposteriori_P_counts);
    DELETE_IF(m_aposteriori_NP_counts);
    DELETE_IF(m_aposteriori_P);
    DELETE_IF(m_aposteriori_NP);

    ClassifierBase::resetModel();

    return true;
}

// Private API

void ClassifierMLkNN_CPU::computeMinMaxValues() {
    std::fill(m_fmax, m_fmax + m_num_attributes, -VALUE_MAX);
    std::fill(m_fmin, m_fmin + m_num_attributes, VALUE_MAX);

    // compute min and max values for scaling
    for(unsigned int i = 0; i<m_num_instances; i++) {
        for(unsigned int j = 0; j<m_num_attributes; j++) {
            value_t val = m_data_X[i * m_num_attributes + j];
            
            if(val > m_fmax[j])
                m_fmax[j] = val;

            if(val < m_fmin[j])
                m_fmin[j] = val;
        }
    }
}

void ClassifierMLkNN_CPU::computeAprioriProb() {
    // Compute prior probabilities

    for(unsigned int i = 0; i<m_num_labels; i++) {
        unsigned int count = 0;

        for(unsigned int j = 0; j<m_num_instances; j++) {
            if(m_data_Y[j * m_num_labels + i] == 1) {
                count += 1;
            }
        }

        m_apriori_P[i] = (m_parm_smooth + count) / (2 * m_parm_smooth + m_num_instances);
        m_apriori_NP[i] = 1 - m_apriori_P[i];
    }

    if(m_debug_fp) {
		logDebug("APRIORI P\n");
		for(unsigned int i = 0; i<m_num_labels; i++)
			logDebug("%f ", m_apriori_P[i]);
		logDebug("\n\n");

		logDebug("APRIORI NP\n");
		for(unsigned int i = 0; i<m_num_labels; i++)
			logDebug("%f ", m_apriori_NP[i]);
		logDebug("\n\n");
    }
}

void ClassifierMLkNN_CPU::computeAposterioriProb() {
    // Compute conditional probabilities
    // First phase - counting labels vs number of instances

    for(unsigned int i = 0; i<m_num_instances; i++) {
        // Get neighbours
        kNN(m_data_X, &m_data_X[i * m_num_attributes], m_num_instances, m_num_attributes, m_parm_k, m_nn, true, i);

        // For each label
        for(unsigned int j = 0; j<m_num_labels; j++) {
            // Count the number of neighbours that have this label
            int count = 0;

            for(unsigned int k = 0; k<m_parm_k; k++) {
                if(m_data_Y[m_nn[k].second * m_num_labels + j] == 1)
                    count += 1;
            }

            // Check if current train instance has this label
            // Increase counter (row -> label, col -> count)
            unsigned int offset = (j * (m_parm_k + 1)) + count;

            if(m_data_Y[i * m_num_labels + j] == 1) {
                m_aposteriori_P_counts[offset] += 1;
            } else {
                m_aposteriori_NP_counts[offset] += 1;
            }
        }
    }

    // Second phase - computing probabilities
    for(unsigned int i = 0; i<m_num_labels; i++) {
        // Compute sums for the denominator
        int sum_aposteriori_P = 0;
        int sum_aposteriori_NP = 0;

        for(unsigned int j = 0; j<m_parm_k + 1; j++) {
            unsigned int offset = i * (m_parm_k + 1) + j;

            sum_aposteriori_P += m_aposteriori_P_counts[offset];
            sum_aposteriori_NP += m_aposteriori_NP_counts[offset];
        }

        // Compute final values for conditional probabilities
        for(unsigned int j = 0; j<m_parm_k + 1; j++) {
            unsigned int offset = i * (m_parm_k + 1) + j;

            m_aposteriori_P[offset] = (m_parm_smooth + m_aposteriori_P_counts[offset]) / (m_parm_smooth * (m_parm_k + 1) + sum_aposteriori_P);
            m_aposteriori_NP[offset] = (m_parm_smooth + m_aposteriori_NP_counts[offset]) / (m_parm_smooth * (m_parm_k + 1) + sum_aposteriori_NP);
        }
    }
}

static inline value_t ClassifierMLkNN_Distance(value_t a, value_t b, unsigned int type, value_t fmin, value_t fmax, value_t smin, value_t smax) {
	value_t distance = 0;

	switch(type) {
		case EAttributeNumeric: {
			a = MLKNN_SCALE_VALUE(a, fmin, fmax, smin, smax);
			b = MLKNN_SCALE_VALUE(b, fmin, fmax, smin, smax);
			distance = a - b;

			break;
		}

		case EAttributeNominal: {
			//distance = (a != b) ? 1 : 0;
			distance = ((a - b) == 0) ? 0: 1;

			break;
		}
	}

	return distance;
}

void ClassifierMLkNN_CPU::kNN(value_t* train_X, value_t* test_X, int num_instances, int num_attributes, int k_neighbours, neighbour_t* nn, bool exclude, unsigned int exclude_id) {
    // prepare
    int nn_count = 0;
    int nn_last = -1;

    // find neighbours in the train data
    for(unsigned int i = 0; i<num_instances; i++) {
        // exclude yourself (we might use test instance from train subset)
        if(exclude && (exclude_id == i))
            continue;

        // calculate Euclidean distance
        value_t distance = 0;

        for(int j = 0; j<num_attributes; j++) {
            value_t a = test_X[j];
            value_t b = train_X[i * num_attributes + j];

            value_t x = ClassifierMLkNN_Distance(a, b, m_ftypes[j], m_fmin[j], m_fmax[j], 0.0f, 1.0f);

            distance += x*x;
        }

        distance = std::sqrt(distance);

        // check if we have less neighbors than k-value
        if(nn_count < k_neighbours) {
            nn[nn_count] = std::make_pair(distance, i);
            nn_count += 1;
        } else {
            // otherwise, check if it's our first time populated array and look for max element
            if(nn_last == -1) {
                nn_last = (int)(std::max_element(nn, nn + nn_count, neighbour_comp) - nn);
            }

            // if we found neighbor that is closer that our max, store it by replacing
            if(distance < nn[nn_last].first) {
            	// check if exists other element with the same distance, but higher index
            	for(unsigned int j = 0; j<nn_count; j++) {
            		if((nn[j].first == nn[nn_last].first) && (nn[j].second > nn[nn_last].second))
            			nn_last = j;
            	}

                nn[nn_last] = std::make_pair(distance, i);
                nn_last = (int)(std::max_element(nn, nn + nn_count, neighbour_comp) - nn);
            }
        }
    }

    // Sort the neighbours
    std::sort(nn, nn + nn_count, neighbour_comp);
}
