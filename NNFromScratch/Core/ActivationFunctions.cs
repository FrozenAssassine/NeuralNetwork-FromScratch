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
                case ActivationType.Sigmoid:
                    return 1.0f / (1.0f + MathF.Exp(-x));
                case ActivationType.Relu:
                    return MathF.Max(0.0f, x);
                case ActivationType.Softmax:
                    return MathF.Exp(x) / (1.0f + MathF.Exp(x));
                case ActivationType.TanH:
                    return MathF.Tanh(x);
                case ActivationType.LeakyRelu:
                    return x > 0.0f ? x : 0.01f * x;
                case ActivationType.ELU:
                    return x > 0.0f ? x : (MathF.Exp(x) - 1.0f);
                case ActivationType.Swish:
                    return x / (1.0f + MathF.Exp(-x));
                default:
                    return x;
            }
        }

        public static float[] Softmax(float[] inputs)
        {
            float sum = 0.0f;
            for (int i = 0; i < inputs.Length; i++)
            {
                sum += MathF.Exp(inputs[i]);
            }
            float[] output = new float[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
            {
                output[i] = MathF.Exp(inputs[i]) / sum;
            }
            return output;
        }

        public static float ActivationDeriv(float x, ActivationType activation)
        {
            switch (activation)
            {
                case ActivationType.Sigmoid:
                    return x * (1 - x);
                case ActivationType.Relu:
                    return x > 0.0f ? 1.0f : 0.0f;
                case ActivationType.Softmax:
                    return x * (1.0f - x);
                case ActivationType.TanH:
                    return 1 - x * x;
                case ActivationType.LeakyRelu:
                    return x > 0.0f ? 1.0f : 0.01f;
                case ActivationType.ELU:
                    return x > 0.0f ? 1.0f : (x + 1.0f);
                case ActivationType.Swish:
                    float sigma = 1.0f / (1.0f + MathF.Exp(-x));
                    return sigma * (1.0f + x * (1.0f - sigma));
                default:
                    return 0.0f;
            }
        }
    }
}
