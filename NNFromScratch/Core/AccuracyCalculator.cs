
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
    public void Calculate(float[][] inputs, float[][] desired, int percent = 10)
    {
        int iterations = (int)(inputs.Length * (percent / 100f));
        for (int i = 0; i < iterations; i++)
        {
            if (MathHelper.GetMaximumIndex(nn.FeedForward_CPU(inputs[i])) == MathHelper.GetMaximumIndex(desired[i]))
                correctCounter++;

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
