<!--
<p align="center">
    <img src="path_to_your_logo" height="150px" width="auto" alt="Neural Network Logo">
</p>
-->

<h1 align="center">Neural Network from scratch with CUDA Support</h1>
<div align="center">
    <img src="https://img.shields.io/github/stars/FrozenAssassine/NeuralNetwork-FromScratch?style=flat"/>
    <img src="https://img.shields.io/github/issues-pr/FrozenAssassine/NeuralNetwork-FromScratch?style=flat"/>
    <img src="https://img.shields.io/github/repo-size/FrozenAssassine/NeuralNetwork-FromScratch?style=flat"/>
</div>

## 🤔 What is this project?
This project is a neural network implementation from scratch in C# with CUDA support written in C++. It currently supports Optical Digit Recognition (ODR) trained with 60,000 images and can also perform XOR as a simple initial test.
I also tried more complex image recognition using a cnn, but I was not able to implement it from scratch.

## ❗Info
At the current point I would not recommend this in any production environment, for me it's just a fun project to learn more about CUDA and Neural Networks.
Also I tried to implement Convolution and Pooling layer from scratch, but failed in the back propagation. Currently they are not working in any way😢


## 🛠️ Features
- **Optical Digit Recognition (ODR)**: Trained with the MNIST dataset of 60,000 images.
- **XOR Test**: A simple test to demonstrate the neural network's basic functionality.
- **CUDA Support**: Accelerates neural network training using GPU resources.
- **CUDA or CPU**: Simply switch between CUDA or CPU processing.

## 📎See also 
- [Deep reinforcement learning](https://github.com/FrozenAssassine/DeepReinforcementLearning) from scratch using this project
- [ESP32 & Arduino](https://github.com/FrozenAssassine/NeuralNetwork-Arduino) running XOR-Demo with a simplified version of this project

- [Interactive Demo](https://frozenassassine.de/nn/xor?ref=github) on my website

## 📊 Benchmarks
| Training Details | GPU (CUDA, RTX 3050) | CPU (i9-10900) 
|------------------|----------------------|----------------|
| 54000 images, 28x28x1 (784 inputs, 512 dense, 256 dense, 10 outputs) | 13.813 sec | 44.001 sec

## 🚀 Performance History
### Sequential to true Parallel 📈 ...

The initial Optical Digit Recognition (ODR) implementation, using 28x28 black-and-white images as input with a neural network consisting of 128 and 64 hidden neurons and 10 output neurons, took 2.8 seconds to train on 1000 images.  
To improve performance, I added **Parallel.For** support, which accelerated the training process. Enabling Release mode further optimized the training time, reducing it to around 780ms for 1000 images.   
However, this was not sufficient. I began integrating CUDA support, which proved challenging but significantly reduced the training time. With CUDA, I brought the training time down to 400ms for 1000 images. In the latest build, I achieved a training time of approximately 200ms per 1000 images.   
Overall, this resulted in a 10 times increase in performance.


## 🏗️ Get Started
1. Clone the repository.
2. Ensure you have the necessary dependencies for C# and CUDA development.
   (https://developer.nvidia.com/cuda-downloads)
4. Open the solution file (`.sln`) in Visual Studio.
5. Build and run the project.

## Example code
```cs
//XOR prediction
var nnmodel = NetworkBuilder.Create()
    .Stack(new InputLayer(2))
    .Stack(new DenseLayer(4, ActivationType.Sigmoid))
    .Stack(new OutputLayer(1, ActivationType.Sigmoid))
    .Build(true); //set to false to train on CPU

nnmodel.Summary();

float[][] inputs = new float[][] { new float[] { 0, 0 }, new float[] { 0, 1 }, new float[] { 1, 0 }, new float[] { 1, 1 } };
float[][] desired = new float[][] { new float[] { 0 }, new float[] { 1 }, new float[] { 1 }, new float[] { 0 } };
nnmodel.Train(inputs, desired, 15900, 0.01f, 1000, 100);

var prediction = nnmodel.Predict(new float[] { 0, 0 });
Console.WriteLine("Prediction: " + MathHelper.GetMaximumIndex(prediction));
```
