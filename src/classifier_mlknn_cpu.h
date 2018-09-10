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

#ifndef __CLASSIFIER_MLKNN_CPU_H
#define __CLASSIFIER_MLKNN_CPU_H

class ClassifierMLkNN_CPU: public ClassifierBase {
private:
    // dataset
    value_t* m_data_X;
    int* m_data_Y;

    // knn
    neighbour_t* m_nn;

    // scaling
    value_t* m_fmax;
    value_t* m_fmin;
    unsigned int* m_ftypes;

    // a priori prob
    value_t* m_apriori_P;
    value_t* m_apriori_NP;

    // a posteriori prob
    int* m_aposteriori_P_counts;
    int* m_aposteriori_NP_counts;
    value_t* m_aposteriori_P;
    value_t* m_aposteriori_NP;

private:
    void computeMinMaxValues();
    void computeAprioriProb();
    void computeAposterioriProb();
    void debug();
    
    void kNN(value_t* train_X, value_t* test_X, int num_instances, int num_attributes, int k_neighbours, neighbour_t* nn, bool exclude = false, unsigned int exclude_id = 0);

public:
    ClassifierMLkNN_CPU();
    virtual ~ClassifierMLkNN_CPU();

    virtual bool buildModel(Dataset& train, int parm_k, value_t parm_smooth);
    virtual bool syncModel();
    virtual int* predict(Dataset& test);
    virtual bool resetModel();
};

#endif // __CLASSIFIER_MLKNN_CPU_H
