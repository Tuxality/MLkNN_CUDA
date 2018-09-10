/*
 *  MLkNN Dataset (XML parser)
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
#include "rapidxml.h"

bool Dataset::parse_xml(std::fstream& fp) {
    // read whole file into memory
    size_t size;
    fp.seekg(0, std::ios::end);
    size = fp.tellg();
    fp.seekg(0);

    char* buffer = new char[size + 1];
    fp.read(&buffer[0], size);
    buffer[size] = '\0';

    // parse this thing
    rapidxml::xml_document<> doc;
    doc.parse<0>(buffer);

    // looking for "labels" tag
    rapidxml::xml_node<>* parent_node = doc.first_node("labels");
    if(!parent_node) {
        return false;
    }

    // prepare our labels map
    unsigned int counter = 0;

    // looking for "label" tags within parent node
    rapidxml::xml_node<>* node = parent_node->first_node();

    while(node != nullptr) {
        if(strcmp(node->name(), "label") == 0) {
            // get attribute from "label" node
            for(rapidxml::xml_attribute<>* a = node->first_attribute(); a; a = a->next_attribute()) {
                m_labels_map[counter++] = a->value();
                m_labels_names.push_back(a->value());
            }
        }

        node = node->next_sibling();
    }

    m_labels = m_labels_map.size();

    return true;
}
