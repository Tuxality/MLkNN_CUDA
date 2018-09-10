/*
 *  MLkNN CPU Debugger
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

void ClassifierMLkNN_CPU::debug() {
	// MinMax
	#ifdef __DEBUG__
	{
		printf("CPU MAX:\n");
		for(unsigned int i = 0; i<m_num_attributes; i++)
			printf("%.2f ", m_fmax[i]);

		printf("\n");
		printf("CPU MIN:\n");
		for(unsigned int i = 0; i<m_num_attributes; i++)
			printf("%.2f ", m_fmin[i]);
		printf("\n");
	}
	#endif // __DEBUG__

	// A priori probabilities
	#ifdef __DEBUG__
	{
		printf("CPU APRIORI:\n");
		for(unsigned int i = 0; i<m_num_labels; i++)
			printf("%.2f ", m_apriori_P[i]);

		printf("\n");
	}
	#endif // __DEBUG__

	// A posteriori sums (first phase)
	#ifdef __DEBUG__
	{
		unsigned int len = m_num_labels * (m_parm_k + 1);

		printf("CPU P:\n");
		for(unsigned int i = 0; i<len; i++) {
			printf("%i ", m_aposteriori_P_counts[i]);
		}
		printf("\n\n");

		printf("CPU NP:\n");
		for(unsigned int i = 0; i<len; i++) {
			printf("%i ", m_aposteriori_NP_counts[i]);
		}
		printf("\n");
	}
	#endif // __DEBUG__

	// A posteriori sums (second phase)
	#ifdef __DEBUG__
	printf("CPU LSUM P  %.3i -> %.5i\n", i, sum_aposteriori_P);
	printf("CPU LSUM NP %.3i -> %.5i\n", i, sum_aposteriori_NP);
	#endif // __DEBUG__

	// A posteriori final (second phase)
	#ifdef __DEBUG__
	{
		unsigned int len = m_num_labels * (m_parm_k + 1);

		printf("\nCPU APOSTERIORI P:\n");
		for(unsigned int i = 0; i<len; i++) {
			printf("%.2f ", m_aposteriori_P[i]);
		}
		printf("\n");

		printf("\nCPU APOSTERIORI NP:\n");
		for(unsigned int i = 0; i<len; i++) {
			printf("%.2f ", m_aposteriori_NP[i]);
		}
		printf("\n");
	}
	#endif // __DEBUG__
}
