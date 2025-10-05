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
			//throw "Softmax should be applied to a vector, not a single value.";
        case 3: //Tanh
            return tanhf(x);
        case 4: // Leaky ReLU
            return x > 0.0f ? x : 0.01f * x;
        case 5: // ELU
            return x > 0.0f ? x : (expf(x) - 1.0f);
        case 6: // Swish
            return x / (1.0f + expf(-x));
        default:
            return x;
        }
    }

    __device__ static float ActivationDeriv(float x, int activation) {
        switch (activation) {
        case 0: //sigmoid deriv
            return x * (1 - x);
        case 1: //relu deriv
            return x > 0.0f ? 1.0f : 0.0f;
        case 2: //softmax deriv
            //throw "Softmax should be applied to a vector, not a single value.";
        case 3: //Tanh
            return 1 - powf(tanhf(x), 2);
        case 4: // Leaky ReLU deriv
            return x > 0.0f ? 1.0f : 0.01f;
        case 5: // ELU deriv
            return x > 0.0f ? 1.0f : (x + 1.0f);
        case 6: // Swish deriv
            float sigma = 1.0f / (1.0f + expf(-x));
            return sigma * (1.0f + x * (1.0f - sigma));
        default:
            return 0.0f;
        }
    }
};