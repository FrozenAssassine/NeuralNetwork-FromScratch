/*
* CUDA training code for my neural network!
* Variable declarations starting with gpu are allocated in gpu memory
*/

#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "DenseLayer.h"
#include "InputLayer.h"
#include "OutputLayer.h"
#include "BaseLayer.h"

#include "StopWatch.h"
#include "AccuracyCalculator.h"

#define DEBUG

#define CUDA_CHECK(err, code) \
    if (err != cudaSuccess) { \
        printf("\nCUDA: Error (%s): %s at line %d\n", code, cudaGetErrorString(err), __LINE__); \
        exit(EXIT_FAILURE);\
    }

using namespace std;

const int threadsPerBlock = 32;

BaseLayer** gpu_allLayer = nullptr;
BaseLayer** cpu_allLayer = nullptr;
int allLayerCount;

float * gpu_desiredValues;


char* RepeatString(char s, int n) {
    char* outs = new char[n];
    for (int i = 0; i < n; i++) {
        outs[i] = s;
    }
    return outs;
}


void PrintLayerInfo(BaseLayer * layer, const char* layerName) {
    printf("%s - Weights: %p, Biases: %p, NeuronValues: %p, Errors: %p, Size: %d\n",
        layerName, layer->Weights, layer->Biases, layer->NeuronValues, layer->Errors, layer->Size);
}

void FeedForward() {
    cudaError_t err;

    //first layer can be skipped -> feed forward all hidden and output:
    for (int i = 1; i < allLayerCount; i++) {
        gpu_allLayer[i]->FeedForward(threadsPerBlock);
    }

    err = cudaDeviceSynchronize();
    CUDA_CHECK(err, "Feed Forward Synchronize Threads");
}

extern "C" __declspec(dllexport) void TrainSingle(float* inputs, float* desired, int size, float learningRate) {
    cudaError_t err;

    //copy the next inputs & outputs to the gpu memory
    err = cudaMemcpy(gpu_allLayer[0]->NeuronValues, inputs, cpu_allLayer[0]->Size * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "Memcpy Inputs");
    
    if (gpu_desiredValues == nullptr) {
        err = cudaMalloc(&gpu_desiredValues, cpu_allLayer[allLayerCount - 1]->Size * sizeof(float));
        CUDA_CHECK(err, "Malloc Desired Values");
    }
    err = cudaMemcpy(gpu_desiredValues, desired, cpu_allLayer[allLayerCount - 1]->Size * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "Memcopy Desired Values");

    FeedForward();
    
    for (int i = allLayerCount - 1; i >= 0; i--) {
        gpu_allLayer[i]->Train(threadsPerBlock, gpu_desiredValues, learningRate);
    }

    err = cudaDeviceSynchronize();
    CUDA_CHECK(err, "Sync Training Threads");
}

extern "C" __declspec(dllexport) void Predict(float* data, float* prediction) {

    cudaError_t err = cudaMemcpy(gpu_allLayer[0]->NeuronValues, data, cpu_allLayer[0]->Size * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "Memcpy Inputs for Prediction");

    FeedForward();

    cudaMemcpy(prediction, gpu_allLayer[allLayerCount - 1]->NeuronValues, cpu_allLayer[allLayerCount - 1]->Size * sizeof(float), cudaMemcpyDeviceToHost);
}

extern "C" __declspec(dllexport) void Init(int totalLayers) {
    printf("Training on CUDA is enabled\n");
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    allLayerCount = totalLayers;

    // Allocate memory for cpu_allLayer array on the host
    cpu_allLayer = new BaseLayer * [totalLayers];
    gpu_allLayer = new BaseLayer * [totalLayers];

    // Initialize layers to null
    for (int i = 0; i < totalLayers; ++i) {
        cpu_allLayer[i] = nullptr;
        gpu_allLayer[i] = nullptr;
    }
}

extern "C" __declspec(dllexport) bool CheckCuda() {
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    return deviceCount >= 1;
}

void AllocateLayerMemory(BaseLayer* gpuLayer, BaseLayer* cpuLayer, int prevSize, int size, float* biases, float* weights, float* neuronValues, float* errors) {
    cudaError_t err;

    // Assign host pointers
    cpuLayer->Weights = weights;
    cpuLayer->Biases = biases;
    cpuLayer->Errors = errors;
    cpuLayer->NeuronValues = neuronValues;

    // Allocate GPU memory and check for errors
    err = cudaMalloc(&gpuLayer->Biases, size * sizeof(float));
    CUDA_CHECK(err, "Allocating GPU Biases");

    err = cudaMalloc(&gpuLayer->NeuronValues, size * sizeof(float));
    CUDA_CHECK(err, "Allocating GPU NeuronValues");

    err = cudaMalloc(&gpuLayer->Errors, size * sizeof(float));
    CUDA_CHECK(err, "Allocating GPU Errors");

    if (prevSize != 0) {
        err = cudaMalloc(&gpuLayer->Weights, size * prevSize * sizeof(float));
        CUDA_CHECK(err, "Allocating GPU Weights");
    }
    
    // Copy host data to GPU
    err = cudaMemcpy(gpuLayer->Biases, biases, size * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "Copying Biases to GPU");

    err = cudaMemcpy(gpuLayer->NeuronValues, neuronValues, size * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "Copying NeuronValues to GPU");

    err = cudaMemcpy(gpuLayer->Errors, errors, size * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "Copying Errors to GPU");

    if (prevSize != 0) {
        err = cudaMemcpy(gpuLayer->Weights, weights, size * prevSize * sizeof(float), cudaMemcpyHostToDevice);
        CUDA_CHECK(err, "Copying Weights to GPU");
    }

    //PrintLayerInfo(gpuLayer, "GPU Layer");
    //PrintLayerInfo(cpuLayer, "CPU Layer");
}

void InitNextLayers() {   
    //set the nextlayer for every layer, this needs to be done after all layers are initialized,
    //because otherwise, the pointers to the nextLayer do not exist yet
    
    for (int i = 0; i < allLayerCount; i++) {
        cpu_allLayer[i]->nextLayer = i + 1 > allLayerCount ? nullptr : cpu_allLayer[i + 1];
        gpu_allLayer[i]->nextLayer = i + 1 > allLayerCount ? nullptr : gpu_allLayer[i + 1];
    }
}

void FillLayer(BaseLayer* cpuLayer, BaseLayer* gpuLayer, int layerIndex, int size, int activation) {
    if (layerIndex == allLayerCount - 1) {
        InitNextLayers();
    }

    cpuLayer->Size = gpuLayer->Size = size;
    cpuLayer->Activation = gpuLayer->Activation = activation;

    cpuLayer->previousLayer = layerIndex == 0 ? nullptr : cpu_allLayer[layerIndex - 1];
    gpuLayer->previousLayer = layerIndex == 0 ? nullptr : gpu_allLayer[layerIndex - 1];
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

    gpu_allLayer[layerIndex] = new InputLayer();
    cpu_allLayer[layerIndex] = new InputLayer();

    BaseLayer* cpuLayer = cpu_allLayer[layerIndex];
    BaseLayer* gpuLayer = gpu_allLayer[layerIndex];

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

    gpu_allLayer[layerIndex] = new OutputLayer();
    cpu_allLayer[layerIndex] = new OutputLayer();

    BaseLayer* cpuLayer = cpu_allLayer[layerIndex];
    BaseLayer* gpuLayer = gpu_allLayer[layerIndex];

    FillLayer(cpuLayer, gpuLayer, layerIndex, size, activation);
    AllocateLayerMemory(gpuLayer, cpuLayer, prevSize, size, biases, weights, neuronValues, errors);
}

extern "C" __declspec(dllexport) void TrainFull(
    float* inputX, 
    float* desired, 
    int epochs, 
    int samples, //number of items to train
    int features, //number of items per input item because 2d array is flattened
    int outputs, //number of items per desired item because 2d array is flattened
    float learningRate = 0.1f,
    int loggingInterval = 100,
    int epochInterval = 1, 
    float evaluatePercent = 10)
{
    Stopwatch epochTime = new Stopwatch(true);
    Stopwatch stepTime = new Stopwatch(true);
    AccuracyCalculator accCalc;

    for (int e = 0; e < epochs; e++) {

        epochTime.Start();
        stepTime.Start();
        accCalc.NextEpoch();

        float averageStepTime = 0;

        for (int i = 0; i < samples; i++) {
            float* x = &inputX[i * features];
            float* y = &desired[i * outputs];

            TrainSingle(x, y, features, 0.01f);

            //use this, when each epoch takes longer due to more items that need to compute
            if ((i + 1) % loggingInterval == 0)
            {
                stepTime.Stop();

                averageStepTime += stepTime.ElapsedMilliseconds();
                printf("Epoch %d/%d; %d/%d; (%.3fms)\n", e + 1, epochs, i + 1, samples, stepTime.ElapsedMilliseconds());
                stepTime.Start();
            }
        }

        //print epoch every x epochs (default: 100) => for fast training
        if ((e + 1) % epochInterval == 0)
        {
            accCalc.Calculate(inputX, desired, &Predict, samples, features, outputs, 0);

            printf("%s\n", RepeatString('-', 50));
            printf("Epoch %d took %.3fms;", e + 1, epochTime.ElapsedMilliseconds());
            accCalc.PrintAccuracy();
            if (averageStepTime > 0) {
                printf(" avg(%d)ms/step", (int)averageStepTime / (samples / loggingInterval));
            }
            printf("\n");

            //dont print the last line after training => looks weird :D
            if (e != epochs - 1)
                printf("%s\n", RepeatString('-', 50));
        }
    }
}

extern "C" __declspec(dllexport) void InitDenseLayer(
    int layerIndex,
    int prevSize,
    int size,
    float* biases,
    float* weights,
    float* neuronValues,
    float* errors, 
    int activation) 
{
    gpu_allLayer[layerIndex] = new DenseLayer();
    cpu_allLayer[layerIndex] = new DenseLayer();

    BaseLayer* cpuLayer = cpu_allLayer[layerIndex];
    BaseLayer* gpuLayer = gpu_allLayer[layerIndex];

    FillLayer(cpuLayer, gpuLayer, layerIndex, size, activation);
    AllocateLayerMemory(gpuLayer, cpuLayer, prevSize, size, biases, weights, neuronValues, errors);
}


extern "C" __declspec(dllexport) void Cleanup() {
    //free the memory of every layer from the gpu
    for (int i = 0; i < allLayerCount; i++) {
        BaseLayer* gpuLayer = gpu_allLayer[i];

        if (gpuLayer) {
            cudaFree(gpuLayer->Biases);
            cudaFree(gpuLayer->NeuronValues);
            cudaFree(gpuLayer->Errors);
            cudaFree(gpuLayer->Weights);
        }
    }

    cudaFree(gpu_desiredValues);
    delete[] gpu_allLayer;
    //delete[] cpu_allLayer;

    printf("\nCUDA: Done -> Cleaned Up Memory\n");
}

extern "C" __declspec(dllexport) void DoneTraining() {
    cudaError_t err;

    // Copy updated weights and biases back to the host
    for (int i = 0; i < allLayerCount; i++) {
        BaseLayer* gpuLayer = gpu_allLayer[i];
        BaseLayer* cpuLayer = cpu_allLayer[i];

        //copy back all layers:
        err = cudaMemcpy(cpuLayer->Biases, gpuLayer->Biases, gpuLayer->Size * sizeof(float), cudaMemcpyDeviceToHost);
        CUDA_CHECK(err, "10");
        err = cudaMemcpy(cpuLayer->Errors, gpuLayer->NeuronValues, gpuLayer->Size * sizeof(float), cudaMemcpyDeviceToHost);
        CUDA_CHECK(err, "11");
        err = cudaMemcpy(cpuLayer->NeuronValues, gpuLayer->NeuronValues, gpuLayer->Size * sizeof(float), cudaMemcpyDeviceToHost);
        CUDA_CHECK(err, "12");

        //input layer doesn't have any weights:
        if (i != 0) {
            err = cudaMemcpy(cpuLayer->Weights, gpuLayer->Weights, gpuLayer->Size * gpu_allLayer[i - 1]->Size * sizeof(float), cudaMemcpyDeviceToHost);
            CUDA_CHECK(err, "9");

#ifdef DEBUG
            printf("SIZE: (%d), %d\n", i, (gpuLayer->Size * gpu_allLayer[i - 1]->Size));
#endif // DEBUG
        }
#ifdef DEBUG
        printf("MemoryAdress: %d, %p, %p, %p, %p\n", i, cpuLayer->Weights, cpuLayer->Biases, cpuLayer->Errors, cpuLayer->NeuronValues);
#endif // DEBUG
    }
}