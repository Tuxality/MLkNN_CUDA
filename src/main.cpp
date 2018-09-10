/*
 *  MLkNN Entry Point
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

#define GET_DEFAULT_VALUE(x, dest, src, name, defval, conv_func)                                            \
                                                                                                            \
if(argc >= x) {                                                                                             \
    try {                                                                                                   \
        dest = conv_func(src);                                                                              \
    } catch(std::invalid_argument e) {                                                                      \
        LOGA("Caught exception: Invalid argument given!\n");                                                \
        dest = defval;                                                                                      \
    }                                                                                                       \
                                                                                                            \
    if(dest <= 0.0f) {                                                                                      \
        LOGA("Warning: %s set to default value (%i) due to the wrong parameter given.\n", name, defval);    \
        dest = defval;                                                                                      \
    }                                                                                                       \
}

int main(int argc, char* argv[]) {
    // [1] Get parameters
    if(argc < 2) {
        LOGA("usage: ./mlknn dataset <k> <smooth> <kCV>\n");
        LOGA(" * note, dataset should be provided without extension\n");

        return 1;
    }

    // Get number of neighbors
    int parm_k = DEFAULT_PARAM_K;
    GET_DEFAULT_VALUE(3, parm_k, argv[2], "k-neighbors", DEFAULT_PARAM_K, std::stoi);

    // Get smoothing value
    value_t parm_smooth = DEFAULT_PARAM_SMOOTHING;
    GET_DEFAULT_VALUE(4, parm_smooth, argv[3], "smoothing", DEFAULT_PARAM_SMOOTHING, std::stof)
    
    // Get kCV value
    int parm_kCV = DEFAULT_PARAM_KCV;
    GET_DEFAULT_VALUE(5, parm_kCV, argv[4], "kCV", DEFAULT_PARAM_KCV, std::stoi)

    // Get dataset name
    std::string parm_dataset = argv[1];

    // [2] Read the dataset
    DatasetManager dataset(parm_kCV, parm_dataset);
    if(!dataset.is_loaded()) {
        LOGA("Bailing out, dataset not loaded.\n");
        return 1;
    }

    // [3] Prepare...
    LOGA("Starting with params:\n");
    LOGA(" dataset = %s\n", parm_dataset.c_str());
    LOGA(" k       = %i\n", parm_k);
    LOGA(" smooth  = %.2f\n", parm_smooth);
    LOGA(" kCV     = %i\n", parm_kCV);
    LOGA("\n");

    Metrics::results_t results[EMethodsCount];

    std::string methods[EMethodsCount] = {
        "CPU",
        "GPU"
    };

    Metrics::prepare(&results[EMethodsFirst], &methods[EMethodsFirst], EMethodsCount);

    Timer t;
    ClassifierMLkNN_GPU::Init();

    // [4] Go, go, go!

    for(unsigned int i = 0; i<parm_kCV; i++) {
        // Common stuff
        printf("-> %i / %i kCV\n", i + 1, parm_kCV);

        Dataset& train = dataset.get_train(i);
        Dataset& test = dataset.get_test(i);
        unsigned int method = 0;

        int* predict;

        // ML-kNN (CPU)
        LOGA(" -> %s\n", methods[method].c_str());
        t.reset();
        ClassifierMLkNN_CPU mlknn_cpu;
        //mlknn_cpu.enableDebug(train.get_relation(), i + 1, "CPU");
        if(!mlknn_cpu.buildModel(train, parm_k, parm_smooth))
        	return 1;
        mlknn_cpu.syncModel();
        predict = mlknn_cpu.predict(test);
        Metrics::populate(&results[method++], test, predict, t);
        delete[] predict;

        // ML-kNN (GPU)
        LOGA(" -> %s\n", methods[method].c_str());
        t.reset();
        ClassifierMLkNN_GPU mlknn_gpu;
        //mlknn_gpu.enableDebug(train.get_relation(), i + 1, "GPU");
        if(!mlknn_gpu.buildModel(train, parm_k, parm_smooth))
        	return 1;
        mlknn_gpu.syncModel();
        predict = mlknn_gpu.predict(test);
        Metrics::populate(&results[method++], test, predict, t);
        delete[] predict;
    }

    Metrics::calculate(&results[EMethodsFirst], EMethodsCount, parm_kCV);

    // [7] Write results...
    LOGA("\n");
    Metrics::print(&results[EMethodsFirst], EMethodsCount);

    ClassifierMLkNN_GPU::Deinit();

    return 0;
}

#undef GET_DEFAULT_VALUE
