/*
 *  MLkNN GPU Implementation
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

#ifndef __CLASSIFIER_MLKNN_GPU_H
#define __CLASSIFIER_MLKNN_GPU_H

typedef struct {
	void* ptr;
	size_t size;
} cudaAlloc_t;

class ClassifierMLkNN_GPU: public ClassifierBase {
private:
	// dataset
	value_t* m_data_X;
    int* m_data_Y;

    // scaling
    value_t* m_fmax;
    value_t* m_fmin;
    unsigned int* m_ftypes;

    // a priori prob
    unsigned int* m_apriori_sums;
    value_t* m_apriori_P;
    value_t* m_apriori_NP;

    // a posteriori prob
    value_t* m_aposteriori_dist;
    int* m_aposteriori_indices;
    unsigned int* m_aposteriori_P_counts;
    unsigned int* m_aposteriori_NP_counts;
    unsigned int* m_aposteriori_P_sums;
    unsigned int* m_aposteriori_NP_sums;
    value_t* m_aposteriori_P;
    value_t* m_aposteriori_NP;

    // computing precisely memory usage
    std::vector<cudaAlloc_t> m_mem_alloc;
    size_t m_mem_alloc_usage;

private:
    void computeMinMaxValues();
    void computeAprioriProb();
    void computeAposterioriProb();
    void debug();

    bool checkMemoryUsage(Dataset& train, int parm_k);
    template<typename T> void gpuMemset(T* A, int N, T val);
    void gpuVFPGarciaKNN(value_t* d_query, unsigned int query_row, int* d_indices, value_t* d_dist, bool exclude = false);
    bool gpuMemUsageGlobal(value_t& total, value_t& used, value_t divider = 1);
    value_t gpuMemUsageLocal(EMemoryUnit& type);
    void printMemUsage(std::string name);
    void cudaMalloc(void** ptr, size_t size);
    void cudaFree(void* ptr);

public:
    ClassifierMLkNN_GPU();
    virtual ~ClassifierMLkNN_GPU();

    virtual bool buildModel(Dataset& train, int parm_k, value_t parm_smooth);
    virtual bool syncModel();
    virtual int* predict(Dataset& test);
    virtual bool resetModel();

    static void Init();
    static void Deinit();
};

#endif // __CLASSIFIER_MLKNN_GPU_H
