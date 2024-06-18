namespace NNFromScratch.Helper;

public class CSVHelper
{
    public static string[][] LoadCSVDataStr(string path, string removeAny = "")
    {
        string[][] data;

        string fileContent = File.ReadAllText(path);
        var lines = fileContent.Split("\n", StringSplitOptions.RemoveEmptyEntries);

        if (lines.Length == 0)
            return null;

        data = new string[lines.Length][];
        //important to skip the first line:
        for (int i = 1; i < lines.Length; i++)
        {
            data[i] = lines[i].Split(",");
        }

        return data;
    }

    public static float[][] LoadCSVData(string path, string removeAny = "")
    {
        float[][] data;

        string fileContent = File.ReadAllText(path);
        var lines = fileContent.Split("\n", StringSplitOptions.RemoveEmptyEntries);

        if (lines.Length == 0)
            return null;

        data = new float[lines.Length][];
        //important to skip the first line:
        for (int i = 1; i < lines.Length; i++)
        {
            if(removeAny.Length == 0)
                data[i] = lines[i].Split(",").Select(x => float.Parse(x)).ToArray();
            else
                data[i] = lines[i].Replace(removeAny, "").Split(",").Select(x => float.Parse(x)).ToArray();
        }

        return data;
    }

    public static int[] GetCSVRow_Int(string path, int row, int columnCount = -1, string removeAny = "")
    {
        int[] data;

        string fileContent = File.ReadAllText(path);
        var lines = fileContent.Split("\n", StringSplitOptions.RemoveEmptyEntries);

        int count = columnCount == -1 ? lines.Length : columnCount;

        if (lines.Length == 0)
            return null;

        data = new int[count];
        //important to skip the first line:
        for (int i = 1; i < count; i++)
        {
            if (removeAny.Length == 0)
                data[i] = int.Parse(lines[i].Split(",")[row]);
            else
                data[i] = int.Parse(lines[i].Replace(removeAny, "").Split(",")[row]);
        }

        return data;
    }
}
