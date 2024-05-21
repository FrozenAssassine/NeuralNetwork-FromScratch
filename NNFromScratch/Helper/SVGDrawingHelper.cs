using NNFromScratch.Core;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNFromScratch.Helper;

internal class SVGDrawingHelper
{
    private int distancePerLayer = 300;
    private int neuronRadius = 50;
    private int yDistPerNeuron = 50;
    private int shiftX = 100;
    private int shiftY = 100;

    public string DrawNeuron(int x, int y, Neuron neuron)
    {
        return $"<circle cx=\"{x}\" cy=\"{y}\" r=\"{neuronRadius}\" stroke=\"blue\" stroke-width=\"5\" fill=\"green\" />" +
            $"<text x=\"{x}\" y=\"{y}\" text-anchor=\"middle\" alignment-baseline=\"middle\" font-size=\"20\" fill=\"black\">{Math.Round(neuron.bias, 5)}</text>";
    }

    public string DrawLayer(int layerIndex, int maxHeight, Neuron[] layer)
    {
        int layerHeight = CalculateMaxHeight(layer);
        int yStart = shiftY + (maxHeight - layerHeight) / 2;
        int xPos = shiftX + (layerIndex * distancePerLayer);

        StringBuilder layerSVG = new StringBuilder();
        for (int i = 0; i < layer.Length; i++)
        {
            layerSVG.AppendLine(DrawNeuron(xPos, yStart + (i * (yDistPerNeuron + neuronRadius * 2)), layer[i]));
        }
        return layerSVG.ToString();
    }

    public string DrawLinks(Neuron[] layer)
    {
        for(int i = 0; i< layer.Length; i++)
        {
            for(int j = 0; j < layer[i].links.Count; j++)
            {

            }
        }
    }

    public int CalculateMaxHeight(NeuronalNetwork nn)
    {
        int max = Math.Max(Math.Max(nn.outputLayer.Length, nn.hiddenLayer.Length), nn.inputLayer.Length);
        return max * (neuronRadius * 2 + yDistPerNeuron) + shiftY;
    }

    public int CalculateMaxHeight(Neuron[] layer)
    {
        return layer.Length * (neuronRadius * 2 + yDistPerNeuron) + shiftY;
    }

    public void Draw(NeuronalNetwork network)
    {
        var height = CalculateMaxHeight(network);

        StringBuilder svg = new StringBuilder();
        svg.AppendLine($"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\r\n<svg width=\"1000\" height=\"{height}\" xmlns=\"http://www.w3.org/2000/svg\">");

        svg.AppendLine(DrawLayer(0, height, network.inputLayer));
        svg.AppendLine(DrawLayer(1, height, network.hiddenLayer));
        svg.AppendLine(DrawLayer(2, height, network.outputLayer));

        svg.AppendLine("</svg>");
        File.WriteAllText("P:\\nn.svg", svg.ToString());
    }
}
