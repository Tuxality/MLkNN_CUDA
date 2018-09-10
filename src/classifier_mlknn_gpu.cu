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

#include "knn.h"

#include <algorithm>

#include "classifier_mlknn_gpu_kernels.h"
#include "classifier_mlknn_gpu_kernels_knn.h"

#define DELETE_IF(x)    if(x)   {			\
							cudaFree(x);	\
							x = nullptr;	\
						}

ClassifierMLkNN_GPU::ClassifierMLkNN_GPU(): ClassifierBase() {
    m_data_X = nullptr;
    m_data_Y = nullptr;

    m_fmax = nullptr;
    m_fmin = nullptr;
    m_ftypes = nullptr;

    m_apriori_sums = nullptr;
    m_apriori_P = nullptr;
    m_apriori_NP = nullptr;

    m_aposteriori_dist = nullptr;
    m_aposteriori_indices = nullptr;
    m_aposteriori_P_counts = nullptr;
    m_aposteriori_NP_counts = nullptr;
    m_aposteriori_P_sums = nullptr;
    m_aposteriori_NP_sums = nullptr;
    m_aposteriori_P = nullptr;
    m_aposteriori_NP = nullptr;

    m_mem_alloc_usage = 0;
}

ClassifierMLkNN_GPU::~ClassifierMLkNN_GPU() {
	resetModel();
}


bool ClassifierMLkNN_GPU::buildModel(Dataset& train, int parm_k, value_t parm_smooth) {
    resetModel();

    if(!ClassifierBase::buildModel(train, parm_k, parm_smooth))
        return false;

    if(!checkMemoryUsage(train, parm_k))
    	return false;

    m_num_instances = train.get_num_instances();
    m_num_attributes = train.get_num_attributes();
    m_num_labels = train.get_num_labels();

    // Allocate memory for dataset
    cudaMalloc((void**)&m_data_X, m_num_instances * m_num_attributes * sizeof(value_t));
    cudaMalloc((void**)&m_data_Y, m_num_instances * m_num_labels * sizeof(int));

    // Allocate memory for min-max scaler
    cudaMalloc((void**)&m_fmax, m_num_attributes * sizeof(value_t));
    cudaMalloc((void**)&m_fmin, m_num_attributes * sizeof(value_t));
    cudaMalloc((void**)&m_ftypes, m_num_attributes * sizeof(value_t));

    // Allocate memory for a priori prob
    cudaMalloc((void**)&m_apriori_sums, m_num_labels * sizeof(unsigned int));
    cudaMalloc((void**)&m_apriori_P, m_num_labels * sizeof(value_t));
    cudaMalloc((void**)&m_apriori_NP, m_num_labels * sizeof(value_t));

    // Allocate memory for a posteriori prob
    cudaMalloc((void**)&m_aposteriori_dist, m_num_instances * m_num_instances * sizeof(value_t));
    cudaMalloc((void**)&m_aposteriori_indices, m_num_instances * m_parm_k * sizeof(int));
    cudaMalloc((void**)&m_aposteriori_P_counts, m_num_labels * (m_parm_k + 1) * sizeof(unsigned int));
    cudaMalloc((void**)&m_aposteriori_NP_counts, m_num_labels * (m_parm_k + 1) * sizeof(unsigned int));
    cudaMalloc((void**)&m_aposteriori_P_sums, m_num_labels * sizeof(unsigned int));
    cudaMalloc((void**)&m_aposteriori_NP_sums, m_num_labels * sizeof(unsigned int));
    cudaMalloc((void**)&m_aposteriori_P, m_num_labels * (m_parm_k + 1) * sizeof(value_t));
    cudaMalloc((void**)&m_aposteriori_NP, m_num_labels * (m_parm_k + 1) * sizeof(value_t));

    // Zeroing memory
    cudaMemset(m_apriori_sums, 0, m_num_labels * sizeof(unsigned int));
    cudaMemset(m_aposteriori_P_counts, 0, m_num_labels * (m_parm_k + 1) * sizeof(unsigned int));
    cudaMemset(m_aposteriori_NP_counts, 0, m_num_labels * (m_parm_k + 1) * sizeof(unsigned int));
    cudaMemset(m_aposteriori_P_sums, 0, m_num_labels * sizeof(unsigned int));
    cudaMemset(m_aposteriori_NP_sums, 0, m_num_labels * sizeof(unsigned int));

    // Copy data (columnwise!)
    cudaMemcpy(m_data_X, train.get_data_X(true), m_num_instances * m_num_attributes * sizeof(value_t), cudaMemcpyHostToDevice);
    cudaMemcpy(m_data_Y, train.get_data_Y(true), m_num_instances * m_num_labels * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(m_ftypes, train.get_attributes_types(), m_num_attributes * sizeof(unsigned int), cudaMemcpyHostToDevice);

    // Compute values for scaling (used by kNN)
    computeMinMaxValues();

    // Compute apriori probabilities
    computeAprioriProb();

    // Compute aposteriori probabilities
    computeAposterioriProb();

    // Allow syncing model!
    m_model_needsync = true;

	#if defined(__DEBUG__) || defined(__PROFILE__)
    printMemUsage("Build");
	#endif

    return true;
}

bool ClassifierMLkNN_GPU::syncModel() {
	if(!m_model_needsync)
		return false;

	// Synchronize both memory transaction and kernel execution
	cudaDeviceSynchronize();

	#ifdef __DEBUG__
    debug();
	#endif // __DEBUG__

	// Free unused memory
	DELETE_IF(m_aposteriori_dist);
	DELETE_IF(m_aposteriori_indices);
	DELETE_IF(m_aposteriori_P_sums);
	DELETE_IF(m_aposteriori_NP_sums);

	#if defined(__DEBUG__) || defined(__PROFILE__)
	printMemUsage("Sync");
	#endif

    return ClassifierBase::syncModel();
}

int* ClassifierMLkNN_GPU::predict(Dataset& test) {
    if(!m_model_built)
        return nullptr;

    // Get query information
    unsigned int num_instances_test = test.get_num_instances();

    // Allocate memory (host side)
    int* h_predict = new int[num_instances_test * m_num_labels]();

    // Allocate memory (device side)
    value_t* test_X;
    int* test_Y;
    int* predict;
    value_t* dist;
    int* indices;
    unsigned int* sums;

    cudaMalloc((void**)&test_X, num_instances_test * m_num_attributes * sizeof(value_t));
    cudaMalloc((void**)&test_Y, num_instances_test * m_num_labels * sizeof(int));
    cudaMalloc((void**)&predict, num_instances_test * m_num_labels * sizeof(int));

    cudaMalloc((void**)&dist, num_instances_test * m_num_instances * sizeof(value_t));
    cudaMalloc((void**)&indices, num_instances_test * m_parm_k * sizeof(int));
    cudaMalloc((void**)&sums, num_instances_test * m_num_labels * sizeof(unsigned int));

    // Copy data (include columnwise data access)
    cudaMemcpy(test_X, test.get_data_X(true), num_instances_test * m_num_attributes * sizeof(value_t), cudaMemcpyHostToDevice);
    cudaMemcpy(test_Y, test.get_data_Y(true), num_instances_test * m_num_labels * sizeof(int), cudaMemcpyHostToDevice);

    // [1] Find neighbours
    gpuVFPGarciaKNN(test_X, num_instances_test, indices, dist, false);

    // [2] Sum labels counts
    {
    	dim3 threadsPerBlock(
    		NTHREADS_X,
    		NTHREADS_Y,
    		1
    	);

    	dim3 blocksPerGrid(
    		(num_instances_test + threadsPerBlock.x - 1) / threadsPerBlock.x,
    		(m_num_labels + threadsPerBlock.y - 1) / threadsPerBlock.y,
    		1
    	);

    	cuComputePrediction_Sums<<<blocksPerGrid, threadsPerBlock>>>(num_instances_test, m_num_labels, m_num_instances, m_parm_k, indices, m_data_Y, sums);
    }

    // [3] Make final prediction by calculating probabilities
    {
    	dim3 threadsPerBlock(
    		NTHREADS_X,
    		NTHREADS_Y,
    		1
    	);

    	dim3 blocksPerGrid(
    		(num_instances_test + threadsPerBlock.x - 1) / threadsPerBlock.x,
    		(m_num_labels + threadsPerBlock.y - 1) / threadsPerBlock.y,
    		1
    	);

    	cuComputePrediction_Final<<<blocksPerGrid, threadsPerBlock>>>(num_instances_test, m_num_labels, m_parm_k, m_apriori_P, m_apriori_NP, m_aposteriori_P, m_aposteriori_NP, sums, predict);
    }

    // Copy results
    cudaMemcpy(h_predict, predict, num_instances_test * m_num_labels * sizeof(int), cudaMemcpyDeviceToHost);

	#if defined(__DEBUG__) || defined(__PROFILE__)
    printMemUsage("Predict");
    LOGA("%12s: %i\n", "Attributes", m_num_attributes);
    LOGA("%12s: %i\n", "Labels", m_num_labels);
    LOGA("%12s: %i\n", "Train", m_num_instances);
    LOGA("%12s: %i\n", "Test", num_instances_test);
	#endif

    // Free memory
    cudaFree(test_X);
    cudaFree(test_Y);
    cudaFree(predict);

    cudaFree(dist);
    cudaFree(indices);
    cudaFree(sums);

    return h_predict;
}

bool ClassifierMLkNN_GPU::resetModel() {
    DELETE_IF(m_data_X);
    DELETE_IF(m_data_Y);

    DELETE_IF(m_fmax);
    DELETE_IF(m_fmin);
    DELETE_IF(m_ftypes);

    DELETE_IF(m_apriori_sums);
    DELETE_IF(m_apriori_P);
    DELETE_IF(m_apriori_NP);

    DELETE_IF(m_aposteriori_dist);
    DELETE_IF(m_aposteriori_indices);
    DELETE_IF(m_aposteriori_P_counts);
    DELETE_IF(m_aposteriori_NP_counts);
    DELETE_IF(m_aposteriori_P_sums);
    DELETE_IF(m_aposteriori_NP_sums);
    DELETE_IF(m_aposteriori_P);
    DELETE_IF(m_aposteriori_NP);

    ClassifierBase::resetModel();

    return true;
}

// Private API

void ClassifierMLkNN_GPU::computeMinMaxValues() {
	// Prepare arrays for scaling
	gpuMemset(m_fmin, m_num_attributes, VALUE_MAX);
	gpuMemset(m_fmax, m_num_attributes, -VALUE_MAX);

	// Compute min and max values for scaling
	{
		int threadsPerBlock = NTHREADS_X;
		int blocksPerGrid = (m_num_instances + threadsPerBlock - 1) / threadsPerBlock;

		for(unsigned int i = 0; i<m_num_attributes; i++) {
			cuComputeMinMaxValues<<<blocksPerGrid, threadsPerBlock>>>(m_fmin, m_fmax, m_data_X, m_num_instances, i);
		}
	}
}

void ClassifierMLkNN_GPU::computeAprioriProb() {
	// compute sums per label
	{
		dim3 threadsPerBlock(
			NTHREADS_X,
			NTHREADS_Y,
			1
		);

		dim3 blocksPerGrid(
			(m_num_instances + threadsPerBlock.x - 1) / threadsPerBlock.x,
			(m_num_labels + threadsPerBlock.y - 1) / threadsPerBlock.y,
			1
		);

		int sharedMemSize = threadsPerBlock.x * threadsPerBlock.y * sizeof(value_t);

		cuComputeAprioriProb_Sums<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(m_data_Y, m_num_instances, m_num_labels, m_apriori_sums);
	}

	// compute a priori probabilities for each label
	{
		int threadsPerBlock = NTHREADS_X;
		int blocksPerGrid = (m_num_labels + threadsPerBlock - 1) / threadsPerBlock;

		cuComputeAprioriProb<<<blocksPerGrid, threadsPerBlock>>>(m_apriori_P, m_apriori_NP, m_apriori_sums, m_parm_smooth, m_num_instances, m_num_labels);
	}
}

void ClassifierMLkNN_GPU::computeAposterioriProb() {
	// [1] Find neighbours (excluding ourself)
	gpuVFPGarciaKNN(m_data_X, m_num_instances, m_aposteriori_indices, m_aposteriori_dist, true);

    // [2] Compute a posteriori counts (first phase)
	{
		dim3 threadsPerBlock(
			NTHREADS_X,
			NTHREADS_Y,
			1
		);

		dim3 blocksPerGrid(
			(m_num_instances + threadsPerBlock.x - 1) / threadsPerBlock.x,
			(m_num_labels + threadsPerBlock.y - 1) / threadsPerBlock.y,
			1
		);

		cuComputeAposterioriProb_Phase1<<<blocksPerGrid, threadsPerBlock>>>(m_data_Y, m_aposteriori_indices, m_aposteriori_P_counts, m_aposteriori_NP_counts, m_num_instances, m_num_labels, m_parm_k);
	}

	// [3] Compute a posteriori sums of counts
	{
		dim3 threadsPerBlock(
			NTHREADS_X,
			NTHREADS_Y,
			1
		);

		dim3 blocksPerGrid(
			((m_parm_k + 1) + threadsPerBlock.x - 1) / threadsPerBlock.x,
			(m_num_labels + threadsPerBlock.y - 1) / threadsPerBlock.y,
			1
		);

		cuComputeAposterioriProb_Phase2_Sums<<<blocksPerGrid, threadsPerBlock>>>(m_num_labels, m_parm_k, m_aposteriori_P_counts, m_aposteriori_NP_counts, m_aposteriori_P_sums, m_aposteriori_NP_sums);
	}

	// [4] Compute a posteriori final values
	{
		dim3 threadsPerBlock(
			NTHREADS_X,
			NTHREADS_Y,
			1
		);

		dim3 blocksPerGrid(
			((m_parm_k + 1) + threadsPerBlock.x - 1) / threadsPerBlock.x,
			(m_num_labels + threadsPerBlock.y - 1) / threadsPerBlock.y,
			1
		);

		cuComputeAposterioriProb_Phase2_Final<<<blocksPerGrid, threadsPerBlock>>>(m_num_labels, m_parm_k, m_parm_smooth, m_aposteriori_P_counts, m_aposteriori_NP_counts, m_aposteriori_P_sums, m_aposteriori_NP_sums, m_aposteriori_P, m_aposteriori_NP);
	}
}

void ClassifierMLkNN_GPU::gpuVFPGarciaKNN(value_t* d_query, unsigned int query_row, int* d_indices, value_t* d_dist, bool exclude) {
	// [1] Compute distances
	dim3 t_16x16(
			16,
			16,
			1
	);

	dim3 g_16x16(
		(query_row%16 != 0)?((query_row / 16) + 1):(query_row / 16),
		(m_num_instances%16 != 0)?((m_num_instances / 16) + 1):(m_num_instances / 16),
		1
	);

	cuComputeDistanceGlobal<<<g_16x16, t_16x16>>>(m_data_X, m_num_instances, d_query, query_row, m_num_attributes, d_dist, m_fmin, m_fmax, m_ftypes, exclude);

	// [2] Sort columns
	dim3 t_256x1(
		256,
		1,
		1
	);

	dim3 g_256x1(
		(query_row%256 != 0)?((query_row / 256) + 1):(query_row / 256),
		1,
		1
	);

	cuInsertionSort<<<g_256x1, t_256x1>>>(d_dist, d_indices, query_row, m_num_instances, m_parm_k);

	// [3] Decrease indexes by one
	int threadsPerBlock = 256;
	int blocksPerGrid = ((query_row * m_parm_k) + threadsPerBlock - 1)/(threadsPerBlock);

	cuPostprocessIndexes<<<blocksPerGrid, threadsPerBlock>>>(d_indices, query_row * m_parm_k);
}

// Common tools

bool ClassifierMLkNN_GPU::checkMemoryUsage(Dataset& train, int parm_k) {
	value_t total, used;
	if(!gpuMemUsageGlobal(total, used))
		return false;

	value_t needed = 0;
	value_t mem_free = (total - used);

	// Useful stuff
	unsigned int num_instances = train.get_num_instances();
	unsigned int num_attributes = train.get_num_attributes();
	unsigned int num_labels = train.get_num_labels();

	// Dataset
	needed += (num_instances * num_attributes * sizeof(value_t));
	needed += (num_instances * num_labels * sizeof(value_t));

	// Distance matrix
	needed += (num_instances * num_instances * sizeof(value_t));

	// kNN indices
	needed += (num_instances * parm_k * sizeof(int));

	// MinMax, scaling
	needed += (num_attributes * sizeof(value_t));
	needed += (num_attributes * sizeof(value_t));
	needed += (num_attributes * sizeof(unsigned int));

	// Apriori
	needed += (num_labels * sizeof(unsigned int));
	needed += (num_labels * sizeof(value_t));
	needed += (num_labels * sizeof(value_t));

	// Aposteriori
	needed += (num_labels * (parm_k + 1) * sizeof(unsigned int));
	needed += (num_labels * (parm_k + 1) * sizeof(unsigned int));
	needed += (num_labels * sizeof(unsigned int));
	needed += (num_labels * sizeof(unsigned int));
	needed += (num_labels * (parm_k + 1) * sizeof(value_t));
	needed += (num_labels * (parm_k + 1) * sizeof(value_t));

	if(needed > mem_free) {
		LOGA("MLkNN_GPU: Memory %.2f MB needed, but only %.2f MB is available!\n", needed, mem_free / 1024 / 1024);
		return false;
	}

	return true;
}

template<typename T> void ClassifierMLkNN_GPU::gpuMemset(T* A, int N, T val) {
	int threadsPerBlock = 1024;
	if(N < threadsPerBlock)
		threadsPerBlock = N;

	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

	cuMemset<T><<<blocksPerGrid, threadsPerBlock>>>(A, N, val);
}

bool ClassifierMLkNN_GPU::gpuMemUsageGlobal(value_t& total, value_t& used, value_t divider) {
	size_t m_total, m_free;
	if(cudaMemGetInfo(&m_free, &m_total) != cudaSuccess)
		return false;

	total = m_total / divider;
	used = (m_total - m_free) / divider;

	return true;
}

value_t ClassifierMLkNN_GPU::gpuMemUsageLocal(EMemoryUnit& type) {
	value_t usage = m_mem_alloc_usage;

	if(usage < 1024) {
		type = EMemoryUnitBytes;
		return usage;
	} else if(usage < 1024 * 1024) {
		type = EMemoryUnitKBytes;
		return usage / 1024;
	} else if(usage < 1024 * 1024 * 1024) {
		type = EMemoryUnitMBytes;
		return usage / 1024 / 1024;
	}

	type = EMemoryUnitGBytes;
	return usage / 1024 / 1024 / 1024;
}

void ClassifierMLkNN_GPU::printMemUsage(std::string name) {
	#if defined(__DEBUG__) || defined(__PROFILE__)
	EMemoryUnit type;
	value_t usage = gpuMemUsageLocal(type);

	printf("%12s: %.2f ", name.c_str(), usage);

	switch(type) {
		case EMemoryUnitBytes: {
			printf("B");
			break;
		}

		case EMemoryUnitKBytes: {
			printf("KB");
			break;
		}

		case EMemoryUnitMBytes: {
			printf("MB");
			break;
		}

		case EMemoryUnitGBytes: {
			printf("GB");
			break;
		}
	}

	printf(" used\n");
	#endif
}

void ClassifierMLkNN_GPU::cudaMalloc(void** ptr, size_t size) {
	::cudaMalloc(ptr, size);

	#if defined(__DEBUG__) || defined(__PROFILE__)
	cudaAlloc_t alloc = {*ptr, size};
	m_mem_alloc.push_back(alloc);
	m_mem_alloc_usage += size;
	#endif
}

void ClassifierMLkNN_GPU::cudaFree(void* ptr) {
	::cudaFree(ptr);

	#if defined(__DEBUG__) || defined(__PROFILE__)
	for(unsigned int i = 0; i<m_mem_alloc.size(); i++) {
		cudaAlloc_t& alloc = m_mem_alloc.at(i);
		if(alloc.ptr == ptr) {
			m_mem_alloc_usage -= alloc.size;
			m_mem_alloc.erase(m_mem_alloc.begin() + i);
		}
	}
	#endif
}

void ClassifierMLkNN_GPU::Init() {
	cudaSetDevice(0);
	cudaDeviceSynchronize();
}

void ClassifierMLkNN_GPU::Deinit() {
	cudaDeviceReset();
}

