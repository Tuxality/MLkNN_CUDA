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

#ifndef __CLASSIFIER_H
#define __CLASSIFIER_H

class ClassifierBase {
protected:
    unsigned int m_num_instances;
    unsigned int m_num_attributes;
    unsigned int m_num_labels;

    int m_parm_k;
    value_t m_parm_smooth;
    bool m_model_built;
    bool m_model_needsync;

    FILE* m_debug_fp;

protected:
    void logDebug(const char* fmt, ...);

public:
    ClassifierBase();
    virtual ~ClassifierBase();

    virtual bool enableDebug(std::string dataset_name, int current_fold, std::string method_name);
    virtual bool disableDebug();

    virtual bool buildModel(Dataset& train, int parm_k, value_t parm_smooth);
    virtual bool syncModel();
    virtual int* predict(Dataset& test) = 0;
    virtual bool resetModel();

    unsigned int getNumAttributes();
    unsigned int getNumLabels();
    bool isModelBuilt();
};

#endif // __CLASSIFIER_H
