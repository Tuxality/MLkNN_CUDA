/*
 *  MLkNN Dataset (ARFF parser)
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

#include <cstring>
#include <algorithm>
#include <sstream>

#define CMD(x)          if(token == x)
#define CMD_(x)         else CMD(x)
#define CMD_UNKNOWN()   else 

template<typename T> void dump_test(const char* fmt, T* ptr, unsigned int row, unsigned int col, bool columnwise) {
    for(unsigned int i = 0; i<row; i++) {
        for(unsigned int j = 0; j<col; j++) {
            unsigned int pos = i * col + j;

            if(columnwise)
                pos = j * row + i;

            printf(fmt, ptr[pos]);
        }

        printf("\n");
    }
}

typedef struct {
	unsigned int index;
	value_t value;
} sparse_value_t;

typedef struct {
	bool is_label;
	bool is_string;
	unsigned int id;
} attribute_label_t;

bool Dataset::parse_arff(std::fstream& fp) {
    // for holding temporary data and state
    std::vector< std::vector<value_t> > tmp;
    std::vector<std::string> tmp_types;
    std::string line, token;
    bool inData = false;

    // for handling sparse matrix
    std::vector< std::vector<sparse_value_t> > tmp_sparse;
    bool sparse = false;

    // for handling class labels interleaved with features also for handling strings
    std::vector<attribute_label_t> tmp_attribute_label;
    unsigned int tmp_counter_label = 0;
    unsigned int tmp_counter_attribute = 0;

    while(getline(fp, line)) {
        if(inData) {
            std::vector<value_t> v;
            std::vector<sparse_value_t> v_sparse;

            // first instance, lets check if it is a sparse matrix
            if(m_instances == 0) {
            	size_t bracket_left = line.find_first_of("{");
            	size_t bracket_right = line.find_first_of("}");

            	if((bracket_left != std::string::npos) && (bracket_right != std::string::npos))
            		sparse = true;
            }

            if(!sparse) {
            	// dense matrix handling

            	// prepare sstream
            	std::stringstream ss(line);

            	// read sstream, comma separated
            	while(std::getline(ss, token, ',')) {
            		// convert our lovely values to floats (including class label)
            		value_t val = std::atof(token.c_str());
                	v.push_back(val);
            	}

            	// push our vector to the dataset container
            	tmp.push_back(v);
            } else {
            	// sparse matrix handling
            	size_t bracket_left = line.find_first_of("{");
            	line.erase(bracket_left, 1);
            	size_t bracket_right = line.find_first_of("}");
            	line.erase(bracket_right, 1);

            	// prepare sstream
            	std::stringstream ss(line);

            	// read sstream, comma separated
            	while(std::getline(ss, token, ',')) {
            		// convert our lovely values to floats (including class label)
            		std::string index = token.substr(0, token.find_first_of(' '));
            		std::string value = token.substr(token.find_first_of(' ') + 1);

            		sparse_value_t sparse_value;
            		sparse_value.index = std::atoi(index.c_str());
            		sparse_value.value = std::atof(value.c_str());

            		v_sparse.push_back(sparse_value);
            	}

            	// push our vector to the dataset container
            	tmp_sparse.push_back(v_sparse);
            }

            m_instances += 1;

            continue;
        }

        // just in case as sstream is not reliable...
        token = "";

        std::stringstream ss(line);
        ss >> token;

        // empty line
        if(token.length() == 0)
            continue;
        
        // comment
        if(token[0] == '%')
            continue;

        CMD("@relation") {
            ss >> m_relation;
        }

        CMD_("@attribute") {
            // name of attribute
        	std::string attribute_name;
            ss >> attribute_name;

            // type of attribute
            std::string attribute_type;
            ss >> attribute_type;
            tmp_types.push_back(attribute_type);

            // determine whether it is a feature or label
            attribute_label_t st = {false, false, 0};

            for(unsigned int i = 0; i<m_labels_names.size(); i++) {
            	if(m_labels_names.at(i) == attribute_name) {
            		st.is_label = true;
            		break;
            	}
            }

            // TODO: if nominal (starts with '{'), determine values
            //       if strings are found (split by ',')
            //       create a map

            if(st.is_label) {
            	st.id = tmp_counter_label++;
            	tmp_attribute_label.push_back(st);
            } else {
            	m_attributes += 1;
            	st.id = tmp_counter_attribute++;
            	tmp_attribute_label.push_back(st);
            }
        }

        CMD_("@data") {
            inData = true;
            //m_attributes -= m_labels;

            // TODO: handle string attributes
        }

        CMD_UNKNOWN() {
            LOGA("Unknown tag \'%s\'!\n", token.c_str());
            fp.close();
            return false;
        }
    }

    // postprocessing attribute types
    m_attributes_types = new unsigned int[m_attributes];
    bool unsupported = false;

    for(unsigned int i = 0; i<m_attributes; i++) {
    	std::string type = tmp_types.at(i);
    	unsigned int att_type = EAttributeDefault;

    	std::transform(type.begin(), type.end(), type.begin(), tolower);

    	if(type == "numeric") {
    		att_type = EAttributeNumeric;
    	} else if(type[0] == '{') {
    		att_type = EAttributeNominal;
    	} else if(type == "string") {
    		att_type = EAttributeString;
    		unsupported = true;
    	} else if(type == "date") {
    		att_type = EAttributeDate;
    		unsupported = true;
    	}

    	m_attributes_types[i] = att_type;
    }

    if(unsupported)
    	return false;

    // postprocessing stuff...
    unsigned int size_X = m_instances * m_attributes;
    m_data_X = new value_t[size_X];
    m_data_X_cw = new value_t[size_X];

    unsigned int size_Y = m_instances * m_labels;
    m_data_Y = new int[size_Y];
    m_data_Y_cw = new int[size_Y];

    if(!sparse) {
        // dense matrix handling
    	for(int i = 0; i<m_instances; i++) {
    		std::vector<value_t>& v = tmp.at(i);

    		for(unsigned int j = 0; j<v.size(); j++) {
    			value_t val = v.at(j);
    			attribute_label_t& st = tmp_attribute_label.at(j);

    			if(st.is_label) {
    				// label
        			unsigned int pos = i * m_labels + st.id;
        			m_data_Y[pos] = (int)val;

        			// columnwise
        			pos = st.id * m_instances + i;
        			m_data_Y_cw[pos] = (int)val;
    			} else {
    				// attribute
        			unsigned int pos = i * m_attributes + st.id;
        			m_data_X[pos] = val;

        			// columnwise
        			pos = st.id * m_instances + i;
        			m_data_X_cw[pos] = val;
    			}
    		}
    	}
    } else {
    	// sparse matrix handling
    	memset(m_data_X, 0, m_instances * m_attributes * sizeof(value_t));
    	memset(m_data_X_cw, 0, m_instances * m_attributes * sizeof(value_t));
    	memset(m_data_Y, 0, m_instances * m_labels * sizeof(int));
    	memset(m_data_Y_cw, 0, m_instances * m_labels * sizeof(int));

    	for(int i = 0; i<tmp_sparse.size(); i++) {
    		std::vector<sparse_value_t>& v = tmp_sparse.at(i);

    		for(int j = 0; j<v.size(); j++) {
    			sparse_value_t& vs = v.at(j);
    			attribute_label_t& st = tmp_attribute_label.at(vs.index);

    			if(st.is_label) {
    				// label
    				unsigned int pos = i * m_labels + st.id;
    				m_data_Y[pos] = (int)vs.value;

    				// columnwise
    				pos = st.id * m_instances + i;
    				m_data_Y_cw[pos] = (int)vs.value;
    			} else {
    				// attribute
    				unsigned int pos = i * m_attributes + st.id;
    				m_data_X[pos] = vs.value;

    				// columnwise
    				pos = st.id * m_instances + i;
    				m_data_X_cw[pos] = vs.value;
    			}
    		}
    	}
    }

    return true;
}
