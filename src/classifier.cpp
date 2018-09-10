/*
 *  MLkNN Classifier Base Implementation
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

#include <cstdio>
#include <cstdarg>

ClassifierBase::ClassifierBase() {
    m_num_instances = 0;
    m_num_attributes = 0;
    m_num_labels = 0;

    m_parm_k = 0;
    m_parm_smooth = 0.0f;
    m_model_built = false;
    m_model_needsync = false;

    m_debug_fp = nullptr;
}

ClassifierBase::~ClassifierBase() {
    resetModel();
    disableDebug();
}

void ClassifierBase::logDebug(const char* fmt, ...) {
	if(!m_debug_fp)
		return;

	va_list va;
	va_start(va, fmt);
	vfprintf(m_debug_fp, fmt, va);
	va_end(va);
}

bool ClassifierBase::enableDebug(std::string dataset_name, int current_fold, std::string method_name) {
	if(m_debug_fp)
		disableDebug();

	char path[1024] = {0};
	snprintf(path, 1024, "dump/%s-%i-%s.txt", dataset_name.c_str(), current_fold, method_name.c_str());
	m_debug_fp = fopen(path, "w");
	if(!m_debug_fp)
		return false;

	return true;
}

bool ClassifierBase::disableDebug() {
	if(!m_debug_fp)
		return false;

	fclose(m_debug_fp);
	m_debug_fp = nullptr;

	return true;
}

bool ClassifierBase::buildModel(Dataset& train, int parm_k, value_t parm_smooth) {
    if(parm_k <= 0)
        return false;

    if(parm_smooth <= 0.0f)
        return false;
    
    m_parm_k = parm_k;
    m_parm_smooth = parm_smooth;

    return true;
}

bool ClassifierBase::syncModel() {
	if(!m_model_needsync)
		return false;

    // Model built!
    m_model_built = true;
    m_model_needsync = false;

    return true;
}

bool ClassifierBase::resetModel() {    
    m_num_instances = 0;
    m_num_attributes = 0;
    m_num_labels = 0;
    m_model_built = false;

    return true;
}

unsigned int ClassifierBase::getNumAttributes() {
    return m_num_attributes;
}

unsigned int ClassifierBase::getNumLabels() {
    return m_num_labels;
}

bool ClassifierBase::isModelBuilt() {
    return m_model_built;
}
