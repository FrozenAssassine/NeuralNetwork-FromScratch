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
#include "LSTMLayer.h"
#include "BaseLayer.h"
//#define DEBUG

#define CUDA_CHECK(err, code) \
    if (err != cudaSuccess) { \
        printf("\nCUDA: Error (%s): %s at line %d\n", code, cudaGetErrorString(err), __LINE__); \
        exit(EXIT_FAILURE);\
    }

const int threadsPerBlock = 64;

BaseLayer** gpu_allLayer = nullptr;
BaseLayer** cpu_allLayer = nullptr;
int allLayerCount;

float * gpu_desiredValues;

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

extern "C" __declspec(dllexport) void Train(float* inputs, float* desired, int size, float learningRate) {
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

//Predictions only work while training on the gpu, otherwise the data is not copied to the gpu memory.
extern "C" __declspec(dllexport) float* Predict(float* data, float* prediction) {

    cudaError_t err = cudaMemcpy(gpu_allLayer[0]->NeuronValues, data, cpu_allLayer[0]->Size * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "Memcpy Inputs for Prediction");

    FeedForward();

    BaseLayer * outLayer = gpu_allLayer[allLayerCount - 1];
    cudaMemcpy(prediction, outLayer->NeuronValues, sizeof(outLayer->Size) * sizeof(float), cudaMemcpyDeviceToHost);
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

extern "C" __declspec(dllexport) void InitLSTMLayer(
    int layerIndex,
    int prevSize,
    int size,
    float* biases,
    float* neuronValues,
    float* errors,
    float* WeightsInput,
    float* WeightsForget,
    float* WeightsOutput,
    float* WeightsCandidate,
    float* CellState,
    float* OutputGate,
    float* ForgetGate,
    float* InputGate,
    float* CandidateCellState,
    float* inputGateGradients,
    float* forgetGateGradients,
    float* outputGradients,
    float* candidateCellGradients
)
{
    LSTMLayer* cpu_layer = new LSTMLayer();
    LSTMLayer* gpu_layer = new LSTMLayer();

    gpu_allLayer[layerIndex] = gpu_layer;
    cpu_allLayer[layerIndex] = cpu_layer;

    FillLayer(cpu_layer, gpu_layer, layerIndex, size, 0);
    AllocateLayerMemory(gpu_layer, cpu_layer, 0, size, biases, 0, neuronValues, errors);

    //allocate lstm memory:

    //all weights
    cudaError_t err = cudaMalloc(&gpu_layer->WeightsInput, size * prevSize * sizeof(float));
    CUDA_CHECK(err, "Allocating GPU WeightsInput");
    err = cudaMemcpy(gpu_layer->WeightsInput, WeightsInput, size * prevSize * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "Copying WeightsInput to GPU");

    err = cudaMalloc(&gpu_layer->WeightsForget, size * prevSize * sizeof(float));
    CUDA_CHECK(err, "Allocating GPU WeightsForget");
    err = cudaMemcpy(gpu_layer->WeightsForget, WeightsForget, size * prevSize * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "Copying WeightsForget to GPU");

    err = cudaMalloc(&gpu_layer->WeightsOutput, size * prevSize * sizeof(float));
    CUDA_CHECK(err, "Allocating GPU WeightsOutput");
    err = cudaMemcpy(gpu_layer->WeightsOutput, WeightsOutput, size * prevSize * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "Copying WeightsOutput to GPU");

    err = cudaMalloc(&gpu_layer->WeightsCandidate, size * prevSize * sizeof(float));
    CUDA_CHECK(err, "Allocating GPU WeightsCandidate");
    err = cudaMemcpy(gpu_layer->WeightsCandidate, WeightsCandidate, size * prevSize * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "Copying WeightsCandidate to GPU");

    //gradients:
    err = cudaMalloc(&gpu_layer->inputGateGradients, size * sizeof(float));
    CUDA_CHECK(err, "Allocating GPU inputGateGradients");
    err = cudaMemcpy(gpu_layer->inputGateGradients, inputGateGradients, size * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "Copying inputGateGradients to GPU");
    
    err = cudaMalloc(&gpu_layer->forgetGateGradients, size * sizeof(float));
    CUDA_CHECK(err, "Allocating GPU forgetGateGradients");
    err = cudaMemcpy(gpu_layer->forgetGateGradients, forgetGateGradients, size * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "Copying forgetGateGradients to GPU");

    err = cudaMalloc(&gpu_layer->outputGradients, size * sizeof(float));
    CUDA_CHECK(err, "Allocating GPU outputGradients");
    err = cudaMemcpy(gpu_layer->outputGradients, outputGradients, size * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "Copying outputGradients to GPU");

    err = cudaMalloc(&gpu_layer->candidateCellGradients, size * sizeof(float));
    CUDA_CHECK(err, "Allocating GPU candidateCellGradients");
    err = cudaMemcpy(gpu_layer->candidateCellGradients, candidateCellGradients, size * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "Copying candidateCellGradients to GPU");

    err = cudaMalloc(&gpu_layer->CellState, size * sizeof(float));
    CUDA_CHECK(err, "Allocating GPU CellState");
    err = cudaMemcpy(gpu_layer->CellState, CellState, size * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "Copying CellState to GPU");

    err = cudaMalloc(&gpu_layer->OutputGate, size * sizeof(float));
    CUDA_CHECK(err, "Allocating GPU OutputGate");
    err = cudaMemcpy(gpu_layer->OutputGate, OutputGate, size * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "Copying OutputGate to GPU");

    err = cudaMalloc(&gpu_layer->ForgetGate, size * sizeof(float));
    CUDA_CHECK(err, "Allocating GPU ForgetGate");
    err = cudaMemcpy(gpu_layer->ForgetGate, ForgetGate, size * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "Copying ForgetGate to GPU");

    err = cudaMalloc(&gpu_layer->InputGate, size * sizeof(float));
    CUDA_CHECK(err, "Allocating GPU InputGate");
    err = cudaMemcpy(gpu_layer->InputGate, InputGate, size * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "Copying InputGate to GPU");
}

extern "C" __declspec(dllexport) void DoneLSTMLayer(int index) {
    cudaError_t err;

    LSTMLayer* gpuLayer = dynamic_cast<LSTMLayer*>(gpu_allLayer[index]);
    LSTMLayer* cpuLayer = dynamic_cast<LSTMLayer*>(cpu_allLayer[index]);

    unsigned long long size = gpuLayer->Size* gpuLayer->previousLayer->Size * sizeof(float);

    err = cudaMemcpy(cpuLayer->WeightsCandidate, gpuLayer->WeightsCandidate, size, cudaMemcpyDeviceToHost);
    CUDA_CHECK(err, "10");
    err = cudaMemcpy(cpuLayer->WeightsForget, gpuLayer->WeightsForget, size, cudaMemcpyDeviceToHost);
    CUDA_CHECK(err, "11");
    err = cudaMemcpy(cpuLayer->WeightsInput, gpuLayer->WeightsInput, size, cudaMemcpyDeviceToHost);
    CUDA_CHECK(err, "12");
    err = cudaMemcpy(cpuLayer->WeightsOutput, gpuLayer->WeightsOutput, size, cudaMemcpyDeviceToHost);
    CUDA_CHECK(err, "12");

    cudaFree(gpuLayer->WeightsCandidate);
    cudaFree(gpuLayer->WeightsForget);
    cudaFree(gpuLayer->WeightsInput);
    cudaFree(gpuLayer->WeightsOutput);
    cudaFree(gpuLayer->CellState);
    cudaFree(gpuLayer->OutputGate);
    cudaFree(gpuLayer->ForgetGate);
    cudaFree(gpuLayer->InputGate);
    cudaFree(gpuLayer->CandidateCellState);
    cudaFree(gpuLayer->inputGateGradients);
    cudaFree(gpuLayer->forgetGateGradients);
    cudaFree(gpuLayer->outputGradients);
    cudaFree(gpuLayer->candidateCellGradients);
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

        //finish LSTM layer:
        if (LSTMLayer* d1 = dynamic_cast<LSTMLayer*>(gpuLayer))
            DoneLSTMLayer(i);

#ifdef DEBUG
        printf("MemoryAdress: %d, %p, %p, %p, %p\n", i, cpuLayer->Weights, cpuLayer->Biases, cpuLayer->Errors, cpuLayer->NeuronValues);
#endif // DEBUG
    }
    Cleanup();
    printf("\nCUDA: Done -> Cleaned Up Memory\n");
}
