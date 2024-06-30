/*
* CUDA training code for my neural network!
* Variable declarations starting with gpu are allocated in gpu memory
*/

#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "BaseLayer.h"
#include "InputLayer.h"
#include "OutputLayer.h"

#define DEBUG

#define CUDA_CHECK(err, code) \
    if (err != cudaSuccess) { \
        printf("\nCUDA: Error (%s): %s at line %d\n", code, cudaGetErrorString(err), __LINE__); \
    }

const int threadsPerBlock = 256;

BaseLayer* gpu_allLayer = { nullptr };
BaseLayer* cpu_allLayer = { nullptr };
int allLayerCount;

float * gpu_desiredValues;


void FeedForward() {
    cudaError_t err;

    //first layer can be skipped -> feed forward all hidden and output:
    for (int i = 1; i < allLayerCount; ++i) {
        gpu_allLayer[i].FeedForward(threadsPerBlock);
    }

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    CUDA_CHECK(err, "2");
}

extern "C" __declspec(dllexport) void Train(float* inputs, float* desired, int size, float learningRate) {
    cudaError_t err;

    //copy the next inputs & outputs to the gpu memory
    err = cudaMemcpy(gpu_allLayer[0].NeuronValues, inputs, cpu_allLayer[0].Size * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "3");

    if (gpu_desiredValues == nullptr) {
        err = cudaMalloc(&gpu_desiredValues, cpu_allLayer[allLayerCount- 1].Size * sizeof(float));
        CUDA_CHECK(err, "50");
    }
    err = cudaMemcpy(gpu_desiredValues, desired, cpu_allLayer[allLayerCount - 1].Size * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "4");

    FeedForward();

    gpu_allLayer[allLayerCount - 1].Train(threadsPerBlock, gpu_desiredValues, learningRate);
    
    for (int i = allLayerCount - 2; i >= 1; i--) {
        //gpu_allLayer[i].Train(threadsPerBlock, gpu_desiredValues, learningRate);
    }
}

extern "C" __declspec(dllexport) void Init(int totalLayers) {
    printf("Training on CUDA is enabled\n");
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    gpu_allLayer = new BaseLayer[totalLayers];
    cpu_allLayer = new BaseLayer[totalLayers];
}

extern "C" __declspec(dllexport) bool CheckCuda() {
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    return deviceCount >= 1;
}

void AllocateLayerMemory(BaseLayer gpuLayer, BaseLayer cpuLayer, int prevSize, int size, float* biases, float* weights, float* neuronValues, float* errors) {
    
    cpuLayer.Weights = weights;
    cpuLayer.Biases = biases;
    cpuLayer.Errors = errors;
    cpuLayer.NeuronValues = neuronValues;

    // Allocate GPU memory
    cudaError_t err;
    err = cudaMalloc(&gpuLayer.Biases, size * sizeof(float));
    CUDA_CHECK(err, "25");
    err = cudaMalloc(&gpuLayer.NeuronValues, size * sizeof(float));
    CUDA_CHECK(err, "26");
    err = cudaMalloc(&gpuLayer.Errors, size * sizeof(float));
    CUDA_CHECK(err, "27");

    if (prevSize != 0) {
        err = cudaMalloc(&gpuLayer.Weights, size * prevSize * sizeof(float));
        CUDA_CHECK(err, "28");
    }
    // Copy initial data to GPU
    err = cudaMemcpy(gpuLayer.Biases, biases, size * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "29");
    err = cudaMemcpy(gpuLayer.NeuronValues, neuronValues, size * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "30");
    err = cudaMemcpy(gpuLayer.Errors, errors, size * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "31");
    if (prevSize != 0) {
        err = cudaMemcpy(gpuLayer.Weights, weights, size * prevSize * sizeof(float), cudaMemcpyHostToDevice);
        CUDA_CHECK(err, "32");
    }

}

void FillLayer(BaseLayer cpuLayer, BaseLayer gpuLayer, int layerIndex, int size, int activation) {
    cpuLayer.Size = gpuLayer.Size = size;
    cpuLayer.Activation = gpuLayer.Activation = activation;
    cpuLayer.previousLayer = gpuLayer.previousLayer = nullptr;
    cpuLayer.nextLayer = gpuLayer.nextLayer = &gpu_allLayer[layerIndex + 1];

    printf("SIZE: %d", cpuLayer.Size);
}

extern "C" __declspec(dllexport) void InitInputLayer(
    int layerIndex,
    int size,
    float* biases,
    float* weights,
    float* neuronValues,
    float* errors,
    int activation)
{
    BaseLayer& cpuLayer = cpu_allLayer[layerIndex] = InputLayer();
    BaseLayer& gpuLayer = gpu_allLayer[layerIndex] = InputLayer();
    
    FillLayer(cpuLayer, gpuLayer, layerIndex, size, activation);
    AllocateLayerMemory(gpuLayer, cpuLayer, 0, size, biases, weights, neuronValues, errors);
}

extern "C" __declspec(dllexport) void InitOutputLayer(
    int layerIndex,
    int prevSize,
    int size,
    float* biases,
    float* weights,
    float* neuronValues,
    float* errors,
    int activation)
{
    BaseLayer& cpuLayer = cpu_allLayer[layerIndex] = InputLayer();
    BaseLayer& gpuLayer = gpu_allLayer[layerIndex] = InputLayer();

    FillLayer(cpuLayer, gpuLayer, layerIndex, size, activation);
    AllocateLayerMemory(gpuLayer, cpuLayer, prevSize, size, biases, weights, neuronValues, errors);
}


extern "C" __declspec(dllexport) void InitHiddenLayer(
    int layerIndex,
    int prevSize,
    int size,
    float* biases,
    float* weights,
    float* neuronValues,
    float* errors, 
    int activation) 
{
    BaseLayer& cpuLayer = cpu_allLayer[layerIndex] = InputLayer();
    BaseLayer& gpuLayer = gpu_allLayer[layerIndex] = InputLayer();

    FillLayer(cpuLayer, gpuLayer, layerIndex, size, activation);
    AllocateLayerMemory(gpuLayer, cpuLayer, prevSize, size, biases, weights, neuronValues, errors);
}


extern "C" __declspec(dllexport) void Cleanup() {
    //free the memory of every layer from the gpu
    for (int i = 0; i < allLayerCount; i++) {
        cudaFree(gpu_allLayer[i].Biases);
        cudaFree(gpu_allLayer[i].NeuronValues);
        cudaFree(gpu_allLayer[i].Errors);
        cudaFree(gpu_allLayer[i].Weights);
    }

    cudaFree(gpu_desiredValues);
    delete[] gpu_allLayer;
    delete[] cpu_allLayer;
}

extern "C" __declspec(dllexport) void DoneTraining() {
    cudaError_t err;

    // Copy updated weights and biases back to the host
    for (int i = 0; i < allLayerCount; i++) {
        BaseLayer& gpuLayer = gpu_allLayer[i];
        BaseLayer& cpuLayer = cpu_allLayer[i];

        //copy back all layers:
        err = cudaMemcpy(cpuLayer.Biases, gpuLayer.Biases, gpuLayer.Size * sizeof(float), cudaMemcpyDeviceToHost);
        CUDA_CHECK(err, "10");
        err = cudaMemcpy(cpuLayer.Errors, gpuLayer.NeuronValues, gpuLayer.Size * sizeof(float), cudaMemcpyDeviceToHost);
        CUDA_CHECK(err, "11");
        err = cudaMemcpy(cpuLayer.NeuronValues, gpuLayer.NeuronValues, gpuLayer.Size * sizeof(float), cudaMemcpyDeviceToHost);
        CUDA_CHECK(err, "12");

        //input layer doesn't have any weights:
        if (i != 0) {
            err = cudaMemcpy(cpuLayer.Weights, gpuLayer.Weights, gpuLayer.Size * gpu_allLayer[i - 1].Size * sizeof(float), cudaMemcpyDeviceToHost);
            CUDA_CHECK(err, "9");

#ifdef DEBUG
            printf("SIZE: (%d), %d\n", i, (gpuLayer.Size * gpu_allLayer[i - 1].Size));
#endif // DEBUG
        }
#ifdef DEBUG
        printf("MemoryAdress: %d, %p, %p, %p, %p\n", i, cpuLayer.Weights, cpuLayer.Biases, cpuLayer.Errors, cpuLayer.NeuronValues);
#endif // DEBUG
    }
    Cleanup();
    printf("\nCUDA: Done -> Cleaned Up Memory\n");
}
