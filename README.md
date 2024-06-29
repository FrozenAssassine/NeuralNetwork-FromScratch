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

## ❗Info
At the current point I would not recommend this in any production environment, for me its just a fun little project to learn more about CUDA and Neural Networks.


## 🛠️ Features
- **Optical Digit Recognition (ODR)**: Trained with the MNIST dataset of 60,000 images.
- **XOR Test**: A simple test to demonstrate the neural network's basic functionality.
- **CUDA Support**: Accelerates neural network training using GPU resources.

## 📊 Benchmarks
| Training Details | GPU (CUDA, RTX 3050) | CPU (i9-10900) |
|------------------|----------------------|----------------|
| 100 images, 150x150x3 (67500 inputs, 1024 hidden, 512 hidden, 6 outputs) | 6.472 seconds | 9.514 seconds |
| 100 images, 150x150x3 (67500 inputs, 2048 hidden, 1024 hidden, 6 outputs) | 6.832 seconds | 19.765 seconds |

  
## 🚀 Get Started
1. Clone the repository.
2. Ensure you have the necessary dependencies for C# and CUDA development.
3. Open the solution file (`.sln`) in Visual Studio.
4. Build and run the project.
