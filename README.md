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
More complex image classification is in progress. I trained it with 2000 rgb images of 150 * 150 pixels and got some ok results.

## ❗Info
At the current point I would not recommend this in any production environment, for me it's just a fun project to learn more about CUDA and Neural Networks.


## 🛠️ Features
- **Optical Digit Recognition (ODR)**: Trained with the MNIST dataset of 60,000 images.
- **XOR Test**: A simple test to demonstrate the neural network's basic functionality.
- **CUDA Support**: Accelerates neural network training using GPU resources.

## 📊 Benchmarks
| Training Details | GPU (CUDA, RTX 3050) | CPU (i9-10900) | (CPU) Ryzen 5 3500U |
|------------------|----------------------|----------------|----------------|
| 100 images, 150x150x3 (67500 inputs, 1024 hidden, 512 hidden, 256 hidden, 6 outputs) | 2.231 sec | 9.514 sec | 34.472 sec
| 100 images, 150x150x3 (67500 inputs, 2048 hidden, 1024 hidden, 6 outputs) | 6.832 sec | 9.426 sec | 31.467 sec

## 🚀 Performance History
The initial Optical Digit Recognition (ODR) implementation, using 28x28 black-and-white images as input with a neural network consisting of 128 and 64 hidden neurons and 10 output neurons, took 2.8 seconds to train on 1000 images.  
To improve performance, I added **Parallel.For** support, which accelerated the training process. Enabling Release mode further optimized the training time, reducing it to around 780ms for 1000 images.   
However, this was not sufficient. I began integrating CUDA support, which proved challenging but significantly reduced the training time. With CUDA, I brought the training time down to 400ms for 1000 images. In the latest build, I achieved a training time of approximately 200ms per 1000 images.   
Overall, this resulted in a 10-fold increase in performance.


## 🏗️ Get Started
1. Clone the repository.
2. Ensure you have the necessary dependencies for C# and CUDA development.
   (https://developer.nvidia.com/cuda-downloads)
4. Open the solution file (`.sln`) in Visual Studio.
5. Build and run the project.
