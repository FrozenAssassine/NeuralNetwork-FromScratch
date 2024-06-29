#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_runtime.h>

#include <chrono>
#include <stdio.h>
#include <iostream>
#include <vector>

//show more ouputs like memory adress etc.
//#define DEBUG

int threadsPerBlock = 1024;

typedef struct Layer {
    float* Biases;
    float* NeuronValues;
    float* Errors;
    float* Weights;
    int Size;
    int Activation;
} Layer;

int allLayersCount;
Layer* gpuLayers;
Layer* cpuLayers;

float* desiredValues;

#define CUDA_CHECK(err, code) \
    if (err != cudaSuccess) { \
        printf("\nCUDA: Error (%s): %s at line %d\n", code, cudaGetErrorString(err), __LINE__); \
    }

__device__ float Activation(float x, int activation) {
    switch (activation) {
    case 0: //sigmoid
        return 1.0f / (1.0f + expf(-x));
    case 1: //relu
        return fmaxf(0.0f, x);
    case 2: //softmax
        return expf(x) / (1.0f + expf(x));
    }
}

__device__ float ActivationDeriv(float x, int activation) {
    switch (activation) {
    case 0: //sigmoid deriv
        return x * (1 - x);
    case 1: //relu deriv
        return x > 0.0f ? 1.0f : 0.0f;
    case 2: //softmax deriv
        return x * (1.0f - x);
    }
}

__global__ void ff_hiddenValues(Layer current, Layer previous, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float sum = 0.0f;
        int index = idx * previous.Size;
        for (int j = 0; j < previous.Size; j++) {
            sum += previous.NeuronValues[j] * current.Weights[index + j];
        }
        current.NeuronValues[idx] = Activation(sum + current.Biases[idx], current.Activation);
    }
}

__global__ void ff_outputValues(Layer output, Layer prev, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float sum = 0.0f;
        int weightIndex = idx * prev.Size;
        for (int j = 0; j < prev.Size; j++) {
            sum += prev.NeuronValues[j] * output.Weights[weightIndex + j];
        }
        output.NeuronValues[idx] = Activation(sum + output.Biases[idx], output.Activation);
    }
}

__global__ void output_WeightsBiases(Layer output, Layer previous, float learningRate, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float derivNeuronVal = learningRate * output.Errors[idx] * ActivationDeriv(output.NeuronValues[idx], output.Activation);
        int weightIndex = idx * previous.Size;

        for (int j = 0; j < previous.Size; j++) {
            atomicAdd(&output.Weights[weightIndex + j], derivNeuronVal * previous.NeuronValues[j]);
        }
        atomicAdd(&output.Biases[idx], learningRate * output.Errors[idx] * ActivationDeriv(output.NeuronValues[idx], output.Activation));
    }
}

__global__ void hidden_ErrorWeight(Layer next, Layer prev, Layer cur, float learningRate, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float err = 0.0f;
        int index = idx * prev.Size;

        for (int j = 0; j < next.Size; j++) {
            err += (next.Errors[j] * next.Weights[j * cur.Size + idx]);
        }
        float error = err * ActivationDeriv(cur.NeuronValues[idx], cur.Activation);
        cur.Errors[idx] = error;

        error *= learningRate;

        for (int j = 0; j < prev.Size; j++) {
            atomicAdd(&cur.Weights[index + j], error * prev.NeuronValues[j]);
        }
        atomicAdd(&cur.Biases[idx], error);
    }
}

__global__ void output_Errors(Layer output, float* desired, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output.Errors[idx] = desired[idx] - output.NeuronValues[idx];
    }
}

float* FeedForward(float* data, int n) {
    cudaError_t err;

    //calculate neuron values for hidden layer:
    for (int i = 1; i < allLayersCount - 1; i++) {
        Layer& prevLayer = gpuLayers[i - 1];
        Layer& currLayer = gpuLayers[i];

        int blocks = (currLayer.Size + threadsPerBlock - 1) / threadsPerBlock;
        ff_hiddenValues<< < blocks, threadsPerBlock >> > (currLayer, prevLayer, currLayer.Size);
    }

    //Compute neuron values for output layer
    Layer& prevLayer = gpuLayers[allLayersCount - 2];
    Layer& outLayer = gpuLayers[allLayersCount - 1];

    int blocks = (outLayer.Size + threadsPerBlock - 1) / threadsPerBlock;
    ff_outputValues<< <blocks, threadsPerBlock >> > (outLayer, prevLayer, outLayer.Size);

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    CUDA_CHECK(err, "2");

    return outLayer.NeuronValues;
}

extern "C" __declspec(dllexport) void Train(float* inputs, float* desiredOutputs, int size, float learningRate) {
    cudaError_t err;
    Layer& outputLayer = gpuLayers[allLayersCount - 1];
    Layer& prevLayer = gpuLayers[allLayersCount - 2];

    //copy the new inputs and outputs to the gpu
    err = cudaMemcpy(gpuLayers[0].NeuronValues, inputs, cpuLayers[0].Size * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "3");

    if (desiredValues == nullptr) {
        err = cudaMalloc(&desiredValues, cpuLayers[allLayersCount - 1].Size * sizeof(float));
        CUDA_CHECK(err, "50");
    }
    err = cudaMemcpy(desiredValues, desiredOutputs, cpuLayers[allLayersCount - 1].Size * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "4");

    // Perform feedforward pass to get the network's output
    FeedForward(gpuLayers[0].NeuronValues, size);

    // Calculate errors for the output layer
    int outputBlocks = (outputLayer.Size + threadsPerBlock - 1) / threadsPerBlock;
    output_Errors << <outputBlocks, threadsPerBlock >> > (gpuLayers[allLayersCount - 1], desiredValues, outputLayer.Size);

    // Update weights and biases for the output layer
    output_WeightsBiases<< <outputBlocks, threadsPerBlock >> > (gpuLayers[allLayersCount - 1], gpuLayers[allLayersCount - 2], learningRate, outputLayer.Size);

    // Backpropagate the errors to the hidden layers
    for (int i = allLayersCount - 2; i >= 1; i--) {
        Layer& currLayer = gpuLayers[i];
        Layer& prevLayer = gpuLayers[i - 1];
        Layer& nextLayer = gpuLayers[i + 1];

        int errorBlocks = (currLayer.Size + threadsPerBlock - 1) / threadsPerBlock;
        hidden_ErrorWeight<< <errorBlocks, threadsPerBlock>> > (nextLayer, prevLayer, currLayer, learningRate, currLayer.Size);
    }

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    CUDA_CHECK(err, "7");
}

extern "C" __declspec(dllexport) void Cleanup() {
    //free the memory of every layer from the gpu
    for (int i = 0; i < allLayersCount; i++) {
        cudaFree(gpuLayers[i].Biases);
        cudaFree(gpuLayers[i].NeuronValues);
        cudaFree(gpuLayers[i].Errors);
        cudaFree(gpuLayers[i].Weights);
    }

    cudaFree(desiredValues);
    delete[] gpuLayers;
}

extern "C" __declspec(dllexport) void DoneTraining() {
    cudaError_t err;

    // Copy updated weights and biases back to the host
    for (int i = 0; i < allLayersCount; i++) {
        Layer& gpuLayer = gpuLayers[i];
        Layer& cpuLayer = cpuLayers[i];
        
        //copy back all layers:
        err = cudaMemcpy(cpuLayer.Biases, gpuLayer.Biases, gpuLayer.Size * sizeof(float), cudaMemcpyDeviceToHost);
        CUDA_CHECK(err, "10");
        err = cudaMemcpy(cpuLayer.Errors, gpuLayer.NeuronValues, gpuLayer.Size * sizeof(float), cudaMemcpyDeviceToHost);
        CUDA_CHECK(err, "11");
        err = cudaMemcpy(cpuLayer.NeuronValues, gpuLayer.NeuronValues, gpuLayer.Size * sizeof(float), cudaMemcpyDeviceToHost);
        CUDA_CHECK(err, "12");

        //input layer doesn't have any weights:
        if (i != 0) {
            err = cudaMemcpy(cpuLayer.Weights, gpuLayer.Weights, gpuLayer.Size * gpuLayers[i - 1].Size * sizeof(float), cudaMemcpyDeviceToHost);
            CUDA_CHECK(err, "9");
            
#ifdef DEBUG
            printf("SIZE: (%d), %d\n", i, (gpuLayer.Size * gpuLayers[i - 1].Size));
#endif // DEBUG
        }
#ifdef DEBUG
        printf("MemoryAdress: %d, %p, %p, %p, %p\n", i, cpuLayer.Weights, cpuLayer.Biases, cpuLayer.Errors, cpuLayer.NeuronValues);
#endif // DEBUG
    }
    Cleanup();
    printf("\nCUDA: Done -> Cleaned Up Memory\n");
}

extern "C" __declspec(dllexport) bool CheckCuda() {
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    return deviceCount >= 1;
}

extern "C" __declspec(dllexport) void Init(int totalLayer) {
    printf("Training on CUDA is enabled\n");
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    threadsPerBlock = prop.maxThreadsPerBlock;

    allLayersCount = totalLayer;

    gpuLayers = new Layer[totalLayer];
    cpuLayers = new Layer[totalLayer];
}

extern "C" __declspec(dllexport) void InitLayer(int layerIndex, int prevSize, int size, float* biases, float* weights, float* values, float* errors, int activation) {
#ifdef DEBUG
    printf("Initializing layer %d with size %d and prevSize %d\n", layerIndex, size, prevSize);
#endif // DEBUG

    Layer& gpuLayer = gpuLayers[layerIndex];
    gpuLayer.Size = size;
    gpuLayer.Activation = activation;

    Layer& cpuLayer = cpuLayers[layerIndex];
    cpuLayer.Size = size;
    cpuLayer.Biases = biases;
    cpuLayer.Errors = errors;
    cpuLayer.NeuronValues = values;
    cpuLayer.Weights = weights;
    cpuLayer.Activation = activation;

#ifdef DEBUG:

    // Print the host pointers to verify they are valid
    printf("Host Biases: %p\n", biases);
    printf("Host Weights: %p\n", weights);
    printf("Host NeuronValues: %p\n", values);
    printf("Host Errors: %p\n", errors);
#endif

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

#ifdef DEBUG:
    // Print the GPU pointers to ensure they are allocated
    printf("GPU Biases: %p\n", gpuLayer.Biases);
    printf("GPU NeuronValues: %p\n", gpuLayer.NeuronValues);
    printf("GPU Errors: %p\n", gpuLayer.Errors);
    printf("GPU Weights: %p\n", gpuLayer.Weights);
#endif

    // Copy initial data to GPU
    err = cudaMemcpy(gpuLayer.Biases, cpuLayer.Biases, size * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "29");
    err = cudaMemcpy(gpuLayer.NeuronValues, cpuLayer.NeuronValues, size * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "30");
    err = cudaMemcpy(gpuLayer.Errors, cpuLayer.Errors, size * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "31");
    if (prevSize != 0) {
        err = cudaMemcpy(gpuLayer.Weights, cpuLayer.Weights, size * prevSize * sizeof(float), cudaMemcpyHostToDevice);
        CUDA_CHECK(err, "32");
    }

#ifdef DEBUG
    printf("Layer %d initialized successfully\n", layerIndex);
#endif // DEBUG
}
