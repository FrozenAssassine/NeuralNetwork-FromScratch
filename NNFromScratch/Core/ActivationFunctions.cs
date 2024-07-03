using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNFromScratch.Core
{
    internal class ActivationFunctions
    {
        public static float Activation(float x, ActivationType activation)
        {
            switch (activation)
            {
                case ActivationType.Sigmoid: //sigmoid
                    return 1.0f / (1.0f + MathF.Exp(-x));
                case ActivationType.Relu: //relu
                    return MathF.Max(0.0f, x);
                case ActivationType.Softmax: //softmax
                    return MathF.Exp(x) / (1.0f + MathF.Exp(x));
                case ActivationType.TanH: //Tanh
                    return MathF.Tanh(x);
                case ActivationType.LeakyRelu: // Leaky ReLU
                    return x > 0.0f ? x : 0.01f * x;
                case ActivationType.ELU: // ELU
                    return x > 0.0f ? x : (MathF.Exp(x) - 1.0f);
                case ActivationType.Swish: // Swish
                    return x / (1.0f + MathF.Exp(-x));
                default:
                    return x;
            }
        }

        public static float ActivationDeriv(float x, ActivationType activation)
        {
            switch (activation)
            {
                case ActivationType.Sigmoid: //sigmoid deriv
                    return x * (1 - x);
                case ActivationType.Relu: //relu deriv
                    return x > 0.0f ? 1.0f : 0.0f;
                case ActivationType.Softmax: //softmax deriv
                    return x * (1.0f - x);
                case ActivationType.TanH: //Tanh
                    return 1 - MathF.Pow(MathF.Tanh(x), 2);
                case ActivationType.LeakyRelu: // Leaky ReLU deriv
                    return x > 0.0f ? 1.0f : 0.01f;
                case ActivationType.ELU: // ELU deriv
                    return x > 0.0f ? 1.0f : (x + 1.0f);
                case ActivationType.Swish: // Swish deriv
                    float sigma = 1.0f / (1.0f + MathF.Exp(-x));
                    return sigma * (1.0f + x * (1.0f - sigma));
                default:
                    return 0.0f;
            }
        }
    }
}
