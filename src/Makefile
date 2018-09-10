# MLkNN on NVIDIA CUDA based GPU
# Written by Przemyslaw 'Tuxality' Skryjomski
#
# For licensing please see README file attached to the archive.
#

CXX=nvcc
LD=nvcc
CFLAGS=-O3 -std=c++11 -D__NO_DEBUG__ -D__NO_PROFILE__ -I. --compile --relocatable-device-code=false -gencode arch=compute_61,code=compute_61 -gencode arch=compute_61,code=sm_61 -x cu
LFLAGS=--cudart static --relocatable-device-code=false -gencode arch=compute_61,code=compute_61 -gencode arch=compute_61,code=sm_61 -link
DEPS=$(wildcard *.h)
SRC_CPP=$(wildcard *.cpp)
SRC_CUDA=$(wildcard *.cu)
OBJECTS=$(SRC_CPP:.cpp=.o) $(SRC_CUDA:.cu=.o)
TARGET=mlknn
TARGET_PARAM=../datasets/Emotions/Emotions

all: $(TARGET)

%.o: %.cpp $(DEPS)
	$(CXX) $(CFLAGS) $< -o $@

%.o: %.cu $(DEPS)
	$(CXX) $(CFLAGS) $< -o $@

$(TARGET): $(OBJECTS)
	$(LD) $(LFLAGS) $^ -o $@

run: $(OBJECTS)
	@./$(TARGET) $(TARGET_PARAM)

clean:
	rm -f $(TARGET)
	rm -f *.o

release:
	tar -zcvf ../MLkNN_CUDA.tar.gz --exclude='.[^/]*' --exclude='*.gz' --exclude='*.o' --exclude='$(TARGET)' ../*

