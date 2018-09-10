/*
 *  MLkNN Dataset
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

Dataset::Dataset(std::string fn): m_loaded(false) {
    cleanup();

    if(fn != "") {
        parse(fn);
    }
}

Dataset::~Dataset() {
    cleanup();
}

void Dataset::cleanup() {
    if(m_loaded) {
    	delete[] m_attributes_types;
        delete[] m_data_X;
        delete[] m_data_Y;
        delete[] m_data_X_cw;
        delete[] m_data_Y_cw;
    }

    m_labels_map.clear();
    m_labels_names.clear();
    m_attributes_names.clear();
    m_labels = 0;

    m_relation = "";
    m_attributes = 0;
    m_instances = 0;

    m_attributes_types = nullptr;

    m_data_X = nullptr;
    m_data_Y = nullptr;
    m_data_X_cw = nullptr;
    m_data_Y_cw = nullptr;

    // reset state
    m_loaded = false;
}

bool Dataset::is_loaded() {
    return m_loaded;
}

std::string Dataset::get_relation() {
	return m_relation;
}

unsigned int Dataset::get_num_instances() {
    return m_instances;
}

unsigned int Dataset::get_num_attributes() {
    return m_attributes;
}

unsigned int Dataset::get_num_labels() {
    return m_labels;
}

unsigned int* Dataset::get_attributes_types() {
	return m_attributes_types;
}

value_t* Dataset::get_data_X(bool columnwise) {
    if(columnwise)
        return m_data_X_cw;

    return m_data_X;
}

int* Dataset::get_data_Y(bool columnwise) {
    if(columnwise)
        return m_data_Y_cw;
    
    return m_data_Y;
}
