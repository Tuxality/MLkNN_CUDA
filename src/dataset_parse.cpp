/*
 *  MLkNN Dataset (Parser)
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

#define DATASET_PARSE(v, ext)   std::string fn_##v = v + ext;                                       \
                                std::fstream fp_##v(fn_##v, std::ios::in);                          \
                                if(!fp_##v.is_open()) {                                             \
                                    LOGA("Cannot open %s file for reading.\n", fn_##v.c_str());     \
                                    return false;                                                   \
                                }                                                                   \
                                                                                                    \
                                status = parse_##v(fp_##v);                                         \
                                fp_##v.close();                                                     \
                                                                                                    \
                                if(!status) {                                                       \
                                    return false;                                                   \
                                }

bool Dataset::parse(std::string fn) {
    return parse(fn, fn);
}

bool Dataset::parse(std::string xml, std::string arff) {
    if(m_loaded) {
        cleanup();
    }

    bool status = true;

    DATASET_PARSE(xml, ".xml");
    DATASET_PARSE(arff, ".arff");

    m_loaded = true;

    return true;
}
