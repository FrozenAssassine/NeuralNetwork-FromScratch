
using NNFromScratch.Helper;
using System;
using System.Diagnostics;

namespace NNFromScratch.Core;

internal class AccuracyCalculator
{
    private int totalCounter = 0;
    private int correctCounter = 0;
    private NeuralNetwork nn;
    public AccuracyCalculator(NeuralNetwork nn)
    {
        this.nn = nn;
    }
    public void Calculate(float[][] inputs, float[][] desired, int startIndex = 0, bool cuda = false)
    {
        float[] prediction = new float[desired[0].Length];
        for (int i = startIndex; i < inputs.Length; i++)
        {
            if (cuda)
            {
                CudaAccel.Predict(inputs[i], prediction);
                if (MathHelper.GetMaximumIndex(prediction) == MathHelper.GetMaximumIndex(desired[i]))
                    correctCounter++;
            }
            else
            {
                if (MathHelper.GetMaximumIndex(nn.FeedForward_CPU(inputs[i])) == MathHelper.GetMaximumIndex(desired[i]))
                    correctCounter++;
            }
            totalCounter++;
        }
    }

    public void PrintAccuracy()
    {
        Console.WriteLine($"Accuracy {MakeAccuracy()}%");
    }
    public double MakeAccuracy()
    {
        return Math.Round(((double)correctCounter / totalCounter) * 100.0, 4);
    }
    public void NextEpoch()
    {
        correctCounter = totalCounter = 0;
    }
}
