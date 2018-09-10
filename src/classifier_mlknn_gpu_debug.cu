/*
 *  MLkNN GPU Debugger
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

/*
 * Debugging stuff
 */

void ClassifierMLkNN_GPU::debug() {
	// MinMax
	#ifdef __DEBUG__
	value_t* fmax = new value_t[m_num_attributes];
	value_t* fmin = new value_t[m_num_attributes];

	cudaMemcpy(fmax, m_fmax, m_num_attributes * sizeof(value_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(fmin, m_fmin, m_num_attributes * sizeof(value_t), cudaMemcpyDeviceToHost);

	printf("GPU MAX:\n");
	for(unsigned int i = 0; i<m_num_attributes; i++)
		printf("%.2f ", fmax[i]);

	printf("\n");
	printf("GPU MIN:\n");
	for(unsigned int i = 0; i<m_num_attributes; i++)
		printf("%.2f ", fmin[i]);
	printf("\n");

	delete[] fmax;
	delete[] fmin;

	#endif // __DEBUG__

	// A priori probabilities
	#ifdef __DEBUG__
	value_t* apriori = new value_t[m_num_labels];
	cudaMemcpy(apriori, m_apriori_P, m_num_labels * sizeof(value_t), cudaMemcpyDeviceToHost);

	printf("GPU APRIORI:\n");
	for(unsigned int i = 0; i<m_num_labels; i++)
		printf("%.2f ", apriori[i]);

	printf("\n");

	delete[] apriori;
	#endif // __DEBUG__

	// A posteriori sums (first phase)
	#ifdef __DEBUG__
	{
		unsigned int len = m_num_labels * (m_parm_k + 1);
		unsigned int* P = new unsigned int[len];
		unsigned int* NP = new unsigned int[len];

		cudaMemcpy(P, m_aposteriori_P_counts, len * sizeof(unsigned int), cudaMemcpyDeviceToHost);
		cudaMemcpy(NP, m_aposteriori_NP_counts, len * sizeof(unsigned int), cudaMemcpyDeviceToHost);

		printf("GPU P:\n");
		for(unsigned int i = 0; i<len; i++) {
			printf("%i ", P[i]);
		}
		printf("\n");

		printf("\n");

		printf("GPU NP:\n");
		for(unsigned int i = 0; i<len; i++) {
			printf("%i ", NP[i]);
		}
		printf("\n");

		delete[] P;
		delete[] NP;
	}
	#endif // __DEBUG__

	// A posteriori sums (second phase)
	#ifdef __DEBUG__
	{
		unsigned int* tmpP = new unsigned int[m_num_labels];
		unsigned int* tmpNP = new unsigned int[m_num_labels];
		cudaMemcpy(tmpP, m_aposteriori_P_sums, m_num_labels * sizeof(unsigned int), cudaMemcpyDeviceToHost);
		cudaMemcpy(tmpNP, m_aposteriori_NP_sums, m_num_labels * sizeof(unsigned int), cudaMemcpyDeviceToHost);

		printf("\n");
		for(unsigned int i = 0; i<m_num_labels; i++) {
			printf("GPU LSUM P  %.3i -> %.5i\n", i, tmpP[i]);
			printf("GPU LSUM NP %.3i -> %.5i\n", i, tmpNP[i]);
		}
		printf("\n");

		delete[] tmpP;
		delete[] tmpNP;
	}
	#endif // __DEBUG__

	// A posteriori final (second phase)
	#ifdef __DEBUG__
	{
		unsigned int len = m_num_labels * (m_parm_k + 1);
		value_t* tmpP = new value_t[len];
		value_t* tmpNP = new value_t[len];
		cudaMemcpy(tmpP, m_aposteriori_P, len * sizeof(value_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(tmpNP, m_aposteriori_NP, len * sizeof(value_t), cudaMemcpyDeviceToHost);

		printf("\nGPU APOSTERIORI P:\n");
		for(unsigned int i = 0; i<len; i++) {
			printf("%.2f ", tmpP[i]);
		}
		printf("\n");

		printf("\nGPU APOSTERIORI NP:\n");
		for(unsigned int i = 0; i<len; i++) {
			printf("%.2f ", tmpNP[i]);
		}
		printf("\n");

		delete[] tmpP;
		delete[] tmpNP;
	}
	#endif // __DEBUG__
}
