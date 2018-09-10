/*
 *  MLkNN Global Header
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

#ifndef __KNN_H
#define __KNN_H

#include <string>
#include <vector>
#include <cfloat>
#include <map>

// Default params
#define DEFAULT_PARAM_K         10
#define DEFAULT_PARAM_SMOOTHING SMOOTHING_LAPLACE
#define DEFAULT_PARAM_KCV       5

#define SMOOTHING_LAPLACE       1.0f

// typedefs
#ifdef __USE_DOUBLE__
typedef double value_t;
#define VALUE_MAX DBL_MAX
#else
typedef float value_t;
#define VALUE_MAX FLT_MAX
#endif // __USE_DOUBLE__

typedef std::pair<value_t, int> neighbour_t;

static bool neighbour_comp(const neighbour_t& l, const neighbour_t& r) {
    return l.first < r.first;
}

enum EMethods {
    EMethodsFirst = 0,
    EMethodsCPU = 0,
    EMethodsGPU,
    EMethodsCount
};

enum EAttributes {
	EAttributeDefault = 0,
	EAttributeNumeric = 0,
	EAttributeNominal,
	EAttributeString,
	EAttributeDate
};

enum EMemoryUnit {
	EMemoryUnitBytes = 0,
	EMemoryUnitKBytes,
	EMemoryUnitMBytes,
	EMemoryUnitGBytes
};

#define MLKNN_SCALE_VALUE(x, Emin, Emax, min, max)    (((x - Emin) / (Emax - Emin)) * (max - min) + min)

// Private headers
#include "debug.h"
#include "timer.h"
#include "dataset.h"
#include "dataset_manager.h"
#include "metrics.h"
#include "classifier.h"
#include "classifier_mlknn_cpu.h"
#include "classifier_mlknn_gpu.h"

#endif // __KNN_H
