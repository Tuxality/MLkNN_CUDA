# MLkNN_CUDA
Source code for Neurocomputing "Speeding up k-Nearest Neighbors Classifier for Large-Scale Multi-Label Learning on GPUs" submission (NEUCOM-D-17-03724)

### Requirements
Following stuff is required in order to compile and execute this project:
 - GNU/Linux based distribution e.g. Arch Linux
 - GCC or compatible clang compiler
 - NVIDIA CUDA Toolkit 8.x or newer (nvcc compiler available in known path)
 - make utility

### Information
To compile the project navigate to the source directory and type *make*. To run an experiment just execute *./mlknn <dataset>* where *<dataset>* refers to the path in which dataset is located. For dataset *Emotions* stored in directory *Emotions* as a dataset parameter just provide *"Emotions/Emotions"*, provided framework will automatically concatenate the prefixes when looking for both ARFF and XML files for the training and testing set (5-CV is assumed). Sample datasets are provided in this package, although more of them are available here in stratified 5-fold CV, Mulan compatible format: http://www.uco.es/kdis/mllresources/

Additionally, a spreadsheet that allows calculating the memory usage on a GPU under certain dataset criteria (instances, features, labels) is available as an offline HTML document with Javascript code embedded.

### License
This code is released under **GPLv3** license.
