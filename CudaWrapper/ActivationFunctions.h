#pragma once

class ActivationFunctions {

public:
    __device__ static float Activation(float x, int activation) {
        switch (activation) {
        case 0: //sigmoid
            return 1.0f / (1.0f + expf(-x));
        case 1: //relu
            return fmaxf(0.0f, x);
        case 2: //softmax
            return expf(x) / (1.0f + expf(x));
        case 3: //Tanh
            return tanhf(x);
        }
    }

    __device__ static float ActivationDeriv(float x, int activation) {
        switch (activation) {
        case 0: //sigmoid deriv
            return x * (1 - x);
        case 1: //relu deriv
            return x > 0.0f ? 1.0f : 0.0f;
        case 2: //softmax deriv
            return x * (1.0f - x);
        case 3: //Tanh
            return 1 - powf(tanhf(x), 2);
        }
    }
};