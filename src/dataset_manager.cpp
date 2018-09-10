/*
 *  MLkNN Dataset Manager
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

DatasetManager::DatasetManager(unsigned int kCV, std::string fn): m_kCV(kCV) {
    cleanup();
    
    if(fn != "") {
        m_loaded = parse(fn);
    }
}

DatasetManager::~DatasetManager() {
    
}

void DatasetManager::cleanup() {
    for(unsigned int i = 0; i<m_train.size(); i++) {
        Dataset* dataset = m_train.at(i);
        delete dataset;
    }

    for(unsigned int j = 0; j<m_test.size(); j++) {
        Dataset* dataset = m_test.at(j);
        delete dataset;
    }

    m_train.clear();
    m_test.clear();

    m_loaded = false;
}

bool DatasetManager::is_loaded() {
    return m_loaded;
}

Dataset& DatasetManager::get_train(unsigned int k) {
    return *m_train.at(k);
}

Dataset& DatasetManager::get_test(unsigned int k) {
    return *m_test.at(k);
}

unsigned int DatasetManager::get_count() {
    return m_train.size();
}
