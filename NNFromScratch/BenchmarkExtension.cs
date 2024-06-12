using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNFromScratch
{
    public class BenchmarkExtension
    {
        public static string BenchmarkWithRam(Action action)
        {
            Stopwatch sw = new Stopwatch();
            long memoryBefore = GC.GetTotalMemory(true);
            sw.Start();
            action?.Invoke();
            sw.Stop();
            long memoryAfter = GC.GetTotalMemory(true);
            long memoryUsage = memoryAfter - memoryBefore;

            return $"{sw.ElapsedMilliseconds}ms ({sw.ElapsedTicks}ticks)" + (memoryUsage / 1000) + "KB";
        }

        public static string Benchmark(Action action)
        {
            Stopwatch sw = new Stopwatch();
            sw.Start();
            action?.Invoke();
            sw.Stop();

            return $"{sw.ElapsedMilliseconds}ms ({sw.ElapsedTicks}ticks)";
        }

        public static void Benchmark(Action action, string name)
        {
            Console.WriteLine(name + " took " + Benchmark(action));
        }
    }
}
