/*
 *  MLkNN GPU Kernels
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

#ifndef __CLASSIFIER_MLKNN_GPU_KERNELS_H
#define __CLASSIFIER_MLKNN_GPU_KERNELS_H

/*
 * Note:
 *  This file should be included only in the CUDA-compatible source code.
 */

// Common stuff

#include <cstdio>
#include <cfloat>

#define NTHREADS_X 128
#define NTHREADS_Y 8

// Source:
//  - NVIDIA CUDA SDK docs
//  - https://github.com/opencv/opencv/blob/master/modules/cudev/include/opencv2/cudev/util/atomic.hpp

typedef unsigned long long uint64;

#ifdef __USE_DOUBLE__
__device__ static void atomicMax(value_t* address, value_t val) {
	if(*address >= val)
		return;

    uint64* address_as_i = (uint64*)address;
    uint64 old = *address_as_i, assumed;

    do {
        assumed = old;

        if(__longlong_as_double(assumed) >= val)
        	break;

        old = atomicCAS(address_as_i, assumed, __double_as_longlong(val));
    } while (assumed != old);
}

__device__ static void atomicMin(value_t* address, value_t val) {
	if(*address <= val)
		return;

    uint64* address_as_i = (uint64*)address;
    uint64 old = *address_as_i, assumed;

    do {
        assumed = old;

        if(__longlong_as_double(assumed) <= val)
        	break;

        old = atomicCAS(address_as_i, assumed, __double_as_longlong(val));
    } while (assumed != old);
}
#else
__device__ static void atomicMax(value_t* address, value_t val) {
    int* address_as_i = (int*)address;
    int old = *address_as_i, assumed;

    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed, __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
}

__device__ static void atomicMin(value_t* address, value_t val) {
    int* address_as_i = (int*)address;
    int old = *address_as_i, assumed;

    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed, __float_as_int(::fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
}
#endif // __USE_DOUBLE__

// Custom CUDA memset
template<typename T> __global__ void cuMemset(T* A, int N, T val) {
	int el = blockIdx.x * blockDim.x + threadIdx.x;
	if(el >= N)
		return;

	A[el] = val;
}

// Computing Min-Max values

__global__ void cuComputeMinMaxValues(value_t* fmin, value_t* fmax, value_t* data_X, unsigned int num_instances, unsigned int attribute) {
	__shared__ value_t sharedMemory_fmax[NTHREADS_X];
	__shared__ value_t sharedMemory_fmin[NTHREADS_X];
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	sharedMemory_fmax[threadIdx.x] = (i < num_instances)?data_X[attribute * num_instances + i]:-FLT_MAX;
	sharedMemory_fmin[threadIdx.x] = (i < num_instances)?data_X[attribute * num_instances + i]:FLT_MAX;

	__syncthreads();

	if(i >= num_instances)
		return;

	for(int s = blockDim.x / 2; s > 0; s >>= 1) {
		if(threadIdx.x < s) {
			if(sharedMemory_fmax[threadIdx.x + s] > sharedMemory_fmax[threadIdx.x])
				sharedMemory_fmax[threadIdx.x] = sharedMemory_fmax[threadIdx.x + s];

			if(sharedMemory_fmin[threadIdx.x + s] < sharedMemory_fmin[threadIdx.x])
				sharedMemory_fmin[threadIdx.x] = sharedMemory_fmin[threadIdx.x + s];
		}

		__syncthreads();
	}

	if(threadIdx.x == 0) {
		atomicMax(&fmax[attribute], sharedMemory_fmax[0]);
		atomicMin(&fmin[attribute], sharedMemory_fmin[0]);
	}
}

// Computing a priori probabilities

__global__ void cuComputeAprioriProb_Sums(int* data_Y, unsigned int num_instances, unsigned int num_labels, unsigned int* sums) {
	extern __shared__ unsigned int sharedMemory[];
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int s_offset = threadIdx.y * blockDim.x + threadIdx.x;
	unsigned int s_offset_zero = threadIdx.y * blockDim.x + 0;

	sharedMemory[s_offset] = ((i < num_instances) && (j < num_labels))?data_Y[j * num_instances + i]:0;

	__syncthreads();

	if((i >= num_instances) || (j >= num_labels))
		return;

	for(int s = blockDim.x / 2; s > 0; s >>= 1) {
		if(threadIdx.x < s) {
			sharedMemory[s_offset] += sharedMemory[s_offset + s];
		}

		__syncthreads();
	}

	if(threadIdx.x == 0) {
		atomicAdd(&sums[j], sharedMemory[s_offset_zero]);
	}
}

__global__ void cuComputeAprioriProb(value_t* P, value_t* NP, unsigned int* sums, value_t smooth, unsigned int num_instances, unsigned int num_labels) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i >= num_labels)
		return;

	P[i] = (smooth + sums[i]) / (2 * smooth + num_instances);
	NP[i] = 1 - P[i];
}

// Computing a posteriori probabilities

__global__ void cuComputeAposterioriProb_Phase1(int* data_Y, int* neighbours, unsigned int* P_counts, unsigned int* NP_counts, unsigned int num_instances, unsigned int num_labels, unsigned int parm_k) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

	if((i >= num_instances) || (j >= num_labels))
		return;

	// counting where i = instance, j = label
	unsigned int count = 0;

	for(unsigned int k = 0; k<parm_k; k++) {
		// columnwise access
		int nn = neighbours[k * num_instances + i];

		if(data_Y[j * num_instances + nn] == 1) {
			count += 1;
		}
	}

	// check if current train instance has this label
	unsigned int offset = j * (parm_k + 1) + count;

	if(data_Y[j * num_instances + i] == 1) {
		atomicAdd(&P_counts[offset], 1);
	} else {
		atomicAdd(&NP_counts[offset], 1);
	}
}

__global__ void cuComputeAposterioriProb_Phase2_Sums(unsigned int num_labels, unsigned int parm_k, unsigned int* P_counts, unsigned int* NP_counts, unsigned int* P_sums, unsigned int* NP_sums) {
	// i - neighbours (k + 1), j - labels
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

	if((i >= (parm_k + 1)) || (j >= num_labels))
		return;

	unsigned int offset = j * (parm_k + 1) + i;
	atomicAdd(&P_sums[j], P_counts[offset]);
	atomicAdd(&NP_sums[j], NP_counts[offset]);
}

__global__ void cuComputeAposterioriProb_Phase2_Final(unsigned int num_labels, unsigned int parm_k, value_t parm_smooth, unsigned int* P_counts, unsigned int* NP_counts, unsigned int* P_sums, unsigned int* NP_sums, value_t* P, value_t* NP) {
	// i - neighbours (k + 1), j - labels
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

	if((i >= (parm_k + 1)) || (j >= num_labels))
		return;

	unsigned int offset = j * (parm_k + 1) + i;
	P[offset] = (parm_smooth + P_counts[offset]) / (parm_smooth * (parm_k + 1) + P_sums[j]);
	NP[offset] = (parm_smooth + NP_counts[offset]) / (parm_smooth * (parm_k + 1) + NP_sums[j]);
}

// Multi-label prediction

__global__ void cuComputePrediction_Sums(unsigned int num_instances, unsigned int num_labels, unsigned int num_instances_train, unsigned int parm_k, int* neighbours, int* data_Y, unsigned int* sums) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

	if((i >= num_instances) || (j >= num_labels))
		return;

	// counting where i = instance, j = label
	unsigned int count = 0;

	for(unsigned int k = 0; k<parm_k; k++) {
		// columnwise access
		int nn = neighbours[k * num_instances + i];

		if(data_Y[j * num_instances_train + nn] == 1) {
			count += 1;
		}
	}

	sums[i * num_labels + j] = count;
}

__global__ void cuComputePrediction_Final(unsigned int num_instances, unsigned int num_labels, unsigned int parm_k, value_t* apriori_P, value_t* apriori_NP, value_t* aposteriori_P, value_t* aposteriori_NP, unsigned int* sums, int* predict) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

	if((i >= num_instances) || (j >= num_labels))
		return;

	unsigned int offset = i * num_labels + j;

	unsigned int prob_offset = j * (parm_k + 1) + sums[offset];
	value_t prob_P = apriori_P[j] * aposteriori_P[prob_offset];
	value_t prob_NP = apriori_NP[j] * aposteriori_NP[prob_offset];

	if(prob_P >= prob_NP) {
		predict[offset] = 1;
	} else {
		predict[offset] = 0;
	}
}

#endif // __CLASSIFIER_MLKNN_GPU_KERNELS_H
