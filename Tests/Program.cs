using Tests.XOR;
using Tests.TestODR;
using Tests.MCSkinCreator;
using Tests.SceneClassification;
using Tests.SortedListCheck;
using Tests.NextSequencePrediction;
using NNFromScratch.Core.Layers;
using Tests.CNN;

public class Program
{
    public static void Main(string[] args)
    {
        Test_CNN.Run();

        //Test_NextSequencePrediction.Run();
        //Test_SortedListCheck.Run();
        //Test_SceneClassification.Run();
        //MinecraftSkinCreator.Run();
        //Test_ODR.Run();
        //Test_XOR.Run();
        //Test_ODR.Run();
    }
}