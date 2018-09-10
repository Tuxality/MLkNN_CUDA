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

#ifndef __DATASET_H
#define __DATASET_H

#include <fstream>

class Dataset {
private:
    bool m_loaded;

private:
    std::map<int, std::string> m_labels_map;
    std::vector<std::string> m_labels_names;
    std::vector<std::string> m_attributes_names;
    unsigned int m_labels;

    std::string m_relation;
    unsigned int m_attributes;
    unsigned int m_instances;

    unsigned int* m_attributes_types;

    value_t* m_data_X;
    int* m_data_Y;

    value_t* m_data_X_cw;
    int* m_data_Y_cw;

private:
    void cleanup();
    bool parse_xml(std::fstream& fp);
    bool parse_arff(std::fstream& fp);

public:
    Dataset(std::string fn = "");
    virtual ~Dataset();

    bool parse(std::string fn);
    bool parse(std::string xml, std::string arff);

    // getter
    bool is_loaded();

    std::string get_relation();
    unsigned int get_num_instances();
    unsigned int get_num_attributes();
    unsigned int get_num_labels();

    unsigned int* get_attributes_types();
    value_t* get_data_X(bool columnwise = false);
    int* get_data_Y(bool columnwise = false);
};

#endif // __DATASET_H
