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

#ifndef __DATASET_MANAGER_H
#define __DATASET_MANAGER_H

class DatasetManager {
private:
    bool m_loaded;
    unsigned int m_kCV;

private:
    std::vector<Dataset*> m_train;
    std::vector<Dataset*> m_test;

private:
    void cleanup();
    bool parse(std::string fn);

public:
    DatasetManager(unsigned int kCV, std::string fn = "");
    ~DatasetManager();

    bool is_loaded();

    Dataset& get_train(unsigned int k);
    Dataset& get_test(unsigned int k);
    unsigned int get_count();
};

#endif // __DATASET_MANAGER_H
