/*
 *  MLkNN GPU Kernels (kNN)
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

#ifndef __CLASSIFIER_MLKNN_GPU_KERNELS_KNN_H
#define __CLASSIFIER_MLKNN_GPU_KERNELS_KNN_H

/*
 * Note:
 *  This file should be included only in the CUDA-compatible source code.
 */

/* Modifed version of kNN-CUDA
 * Source: https://github.com/vincentfpgarcia/kNN-CUDA
 *
 * Last modified by Przemyslaw Skryjomski <skryjomskipl@vcu.edu> 11/15/2017
 * The modifications are
 *      code cleanup
 *      allows excluding item from the training set
 *      scaling values for MLkNN (min-max w/ use of shared memory)
 *      handling numeric and nominal attributes
 *
 * Previously modified by Christopher B. Choy <chrischoy@ai.stanford.edu> 12/23/2016
 * The modifications are
 *      removed texture memory usage
 *      removed split query KNN computation
 *      added feature extraction with bilinear interpolation
 */

// Constants used by the program
#define BLOCK_DIM                      16

/**
  * Computes the distance between two matrix A (reference points) and
  * B (query points) containing respectively wA and wB points.
  *
  * @param A     pointer on the matrix A
  * @param wA    width of the matrix A = number of points in A
  * @param B     pointer on the matrix B
  * @param wB    width of the matrix B = number of points in B
  * @param dim   dimension of points = height of matrices A and B
  * @param AB    pointer on the matrix containing the wA*wB distances computed
  *
  */

__device__ inline value_t ClassifierMLkNN_Distance(value_t a, value_t b, unsigned int type, value_t fmin, value_t fmax, value_t smin, value_t smax) {
	value_t distance = 0;

	switch(type) {
		case EAttributeNumeric: {
			a = MLKNN_SCALE_VALUE(a, fmin, fmax, smin, smax);
			b = MLKNN_SCALE_VALUE(b, fmin, fmax, smin, smax);
			distance = a - b;

			break;
		}

		case EAttributeNominal: {
			distance = (a != b) ? 1 : 0;

			break;
		}
	}

	return distance;
}

__global__ void cuComputeDistanceGlobal(value_t* A, int wA, value_t* B, int wB, int dim, value_t* AB, value_t* fmin, value_t* fmax, unsigned int* ftypes, bool exclude) {
  // Declaration of the shared memory arrays As and Bs used to store the sub-matrix of A and B
  __shared__ value_t shared_A[BLOCK_DIM][BLOCK_DIM];
  __shared__ value_t shared_B[BLOCK_DIM][BLOCK_DIM];
  __shared__ int shared_ATT[BLOCK_DIM][BLOCK_DIM];

  // Sub-matrix of A (begin, step, end) and Sub-matrix of B (begin, step)
  __shared__ int begin_A;
  __shared__ int begin_B;
  __shared__ int step_A;
  __shared__ int step_B;
  __shared__ int end_A;

  // Thread index
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Other variables
  value_t tmp;
  value_t ssd = 0;

  // Loop parameters
  begin_A = BLOCK_DIM * blockIdx.y;
  begin_B = BLOCK_DIM * blockIdx.x;
  step_A  = BLOCK_DIM * wA;
  step_B  = BLOCK_DIM * wB;
  end_A   = begin_A + (dim-1) * wA;

    // Conditions
  int cond0 = (begin_A + tx < wA); // used to write in shared memory
  int cond1 = (begin_B + tx < wB); // used to write in shared memory & to computations and to write in output matrix
  int cond2 = (begin_A + ty < wA); // used to computations and to write in output matrix

  // Loop over all the sub-matrices of A and B required to compute the block sub-matrix

  for (int a = begin_A, b = begin_B; a <= end_A; a += step_A, b += step_B) {
    // Load the matrices from device memory to shared memory; each thread loads one element of each matrix
    if (a/wA + ty < dim){
      shared_A[ty][tx] = (cond0)? A[a + wA * ty + tx] : 0;
      shared_B[ty][tx] = (cond1)? B[b + wB * ty + tx] : 0;
      shared_ATT[ty][tx] = a/wA + ty;
    }
    else{
      shared_A[ty][tx] = 0;
      shared_B[ty][tx] = 0;
      shared_ATT[ty][tx] = -1;
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Compute the difference between the two matrixes; each thread computes one element of the block sub-matrix
    if (cond2 && cond1){
      for (int k = 0; k < BLOCK_DIM; ++k){
    	value_t v_a = shared_A[k][ty];
    	value_t v_b = shared_B[k][tx];

    	int attribute = shared_ATT[k][ty];
    	if(attribute == -1) {
    		tmp = 0;
    	} else {
    		tmp = ClassifierMLkNN_Distance(v_a, v_b, ftypes[attribute], fmin[attribute], fmax[attribute], 0.0f, 1.0f);
    	}

    	ssd += tmp*tmp;
      }
    }

    // Synchronize to make sure that the preceding computation is done before loading two new sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory; each thread writes one element
  if (cond2 && cond1) {
	// Exclude ourselves in the distance computation by setting FLT_MAX
	if(exclude) {
		unsigned int a = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int b = blockIdx.y * blockDim.y + threadIdx.y;

		if(a == b) {
			ssd = FLT_MAX;
		}
	} else {
		ssd = sqrt(ssd);
	}

	/*
	// sanity check, not sure if needed
	unsigned int offset = (begin_A + ty) * wB + begin_B + tx;
	if(offset < dim * wA) {
		AB[(begin_A + ty) * wB + begin_B + tx] = ssd;
	}
	*/

	AB[(begin_A + ty) * wB + begin_B + tx] = ssd;
  }
}


/**
  * Gathers k-th smallest distances for each column of the distance matrix in the top.
  *
  * @param dist        distance matrix
  * @param ind         index matrix
  * @param width       width of the distance matrix and of the index matrix
  * @param height      height of the distance matrix and of the index matrix
  * @param k           number of neighbors to consider
  *
  */

__global__ void cuInsertionSort(value_t *dist, int *ind, int width, int height, int k) {
	  // Variables
	  int l, i, j;
	  value_t *p_dist;
	  int   *p_ind;
	  value_t curr_dist, max_dist;
	  int   curr_row,  max_row;
	  unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;

	  if (xIndex<width){
	    // Pointer shift, initialization, and max value
	    p_dist   = dist + xIndex;
	    p_ind    = ind  + xIndex;
	    max_dist = p_dist[0];
	    p_ind[0] = 1;

	    // Part 1 : sort kth firt elementZ
	    for (l=1; l<k; l++){
	      curr_row  = l * width;
	      curr_dist = p_dist[curr_row];
	      if (curr_dist<max_dist){
	        i=l-1;
	        for (int a=0; a<l-1; a++){
	          if (p_dist[a*width]>curr_dist){
	            i=a;
	            break;
	          }
	        }
	        for (j=l; j>i; j--){
	          p_dist[j*width] = p_dist[(j-1)*width];
	          p_ind[j*width]   = p_ind[(j-1)*width];
	        }
	        p_dist[i*width] = curr_dist;
	        p_ind[i*width]   = l+1;
	      } else {
	        p_ind[l*width] = l+1;
	      }
	      max_dist = p_dist[curr_row];
	    }

	    // Part 2 : insert element in the k-th first lines
	    max_row = (k-1)*width;
	    for (l=k; l<height; l++){
	      curr_dist = p_dist[l*width];
	      if (curr_dist<max_dist){
	        i=k-1;
	        for (int a=0; a<k-1; a++){
	          if (p_dist[a*width]>curr_dist){
	            i=a;
	            break;
	          }
	        }
	        for (j=k-1; j>i; j--){
	          p_dist[j*width] = p_dist[(j-1)*width];
	          p_ind[j*width]   = p_ind[(j-1)*width];
	        }
	        p_dist[i*width] = curr_dist;
	        p_ind[i*width]   = l+1;
	        max_dist             = p_dist[max_row];
	      }
	    }
	  }
}

/**
  * Decreases neighbours indexes by 1.
  *
  * @param dist    distance matrix
  * @param width   width of the distance matrix
  * @param k       number of neighbors to consider
  *
  */

__global__ void cuPostprocessIndexes(int* ind, unsigned int num_items) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= num_items)
    	return;

    ind[tid] -= 1;
}

#endif // __CLASSIFIER_MLKNN_GPU_KERNELS_KNN_H
