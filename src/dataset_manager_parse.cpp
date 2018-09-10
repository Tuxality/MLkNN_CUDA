/*
 *  MLkNN Dataset Manager (Parser)
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

#define DATASET_MANAGER_LOAD(v, k, fmt, ...)    char fmt_##v[256] = {0};                    \
                                                snprintf(fmt_##v, 256, fmt, __VA_ARGS__);   \
                                                                                            \
                                                std::string s_##v = fn + fmt_##v;           \
                                                Dataset* d_##v = new Dataset();             \
                                                                                            \
                                                if(!d_##v->parse(fn, s_##v)) {              \
                                                    cleanup();                              \
                                                    return false;                           \
                                                }                                           \
                                                                                            \
                                                m_##v.push_back(d_##v);

bool DatasetManager::parse(std::string fn) {
    for(int i = 0; i<m_kCV; i++) {
        DATASET_MANAGER_LOAD(train, i, "-train%i", i + 1);
        DATASET_MANAGER_LOAD(test, i, "-test%i", i + 1);
    }

    return true;
}
