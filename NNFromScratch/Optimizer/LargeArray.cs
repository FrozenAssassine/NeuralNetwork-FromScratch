
namespace NNFromScratch.Optimizer
{
    public class LargeArray<N>
    {
        List<N[]> data;
        public static int chunkSize;
        int size;

        public int Length => size;

        public LargeArray(int size)
        {
            this.size = size;
            int chunks = size / chunkSize;
            if (size % chunkSize != 0)
                chunks++;
            data = new List<N[]>();
            for (int i = 0; i < chunks; i++)
                data.Add(new N[chunkSize]);
        }

        public N this[int index]
        {
            get
            {
                if (index >= size)
                    throw new IndexOutOfRangeException();
                return data[index / chunkSize][index % chunkSize];
            }
            set
            {
                if (index >= size)
                    throw new IndexOutOfRangeException();
                data[index / chunkSize][index % chunkSize] = value;
            }
        }

        public N[] Get(int startIndex, int length, bool clone = false)
        {
            int endIndex = startIndex + length - 1;
            if (startIndex < 0 || startIndex >= size || endIndex < 0 || endIndex >= size)
                throw new IndexOutOfRangeException();
            if (length == chunkSize && startIndex % chunkSize == 0)
            {
                if (clone)
                {
                    N[] cloneRes = new N[chunkSize];
                    Array.Copy(data[startIndex / chunkSize], cloneRes, chunkSize);
                    return cloneRes;
                }
                else
                    return data[startIndex / chunkSize];
            }
            N[] res = new N[length];
            int startChunk = startIndex / chunkSize;
            int endChunk = endIndex / chunkSize;
            int startChunkStartPos = startIndex % chunkSize;
            if (startChunk == endChunk)
            {
                Array.Copy(data[startChunk], startChunkStartPos, res, 0, length);
                return res;
            }
            int endChunkEndPos = endIndex % chunkSize;
            int currentResIndex = 0;
            Array.Copy(data[startChunk], startChunkStartPos, res, currentResIndex, chunkSize - startChunkStartPos);
            currentResIndex += startChunkStartPos;
            for (int i = startChunk + 1; i < endChunk; i++)
            {
                Array.Copy(data[i], 0, res, currentResIndex, chunkSize);
                currentResIndex += chunkSize;
            }
            Array.Copy(data[endChunk], 0, res, currentResIndex, endChunkEndPos);
            return res;
        }

        public static explicit operator N[](LargeArray<N> d)
        {
            return d.Get(0, d.size, true);
        }

        public static explicit operator LargeArray<N>(N[] d)
        {
            LargeArray<N> res = new LargeArray<N>(d.Length);
            for (int i = 0; i < d.Length; i++)
            {
                res[i] = d[i];
            }
            return res;
        }
    }
}
