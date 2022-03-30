using MathNet.Numerics.LinearAlgebra;

namespace BCM
{
    class Program
    {
        public static void Main(string[] args)
        {
            // Generator.Generator.GenerateNewMatrix(1, 10000, 5, 0, 1); 
            string filename = "1n1000_d5_m0_std1.csv";
            string path = "D:/University Work/OneDrive - Monash University/5th Year/ENG4702/BCM/TestCases/";
            Matrix<double> input = SparseIO.Reader($"{path}{filename}");

            IEnumerable<double> output = ThreadedBCM.Solver(-input);

            Output(output, $"stream_{filename}");
            return;
        }

        public static double Solution(Matrix<double> input, Matrix<double> sigma)
        {
            return input.PointwiseMultiply(sigma * sigma.Transpose()).RowSums().Sum();
        }

        public static void Output(IEnumerable<double> output, string filename)
        {
            FileStream stream = new FileStream(filename, FileMode.Create);
            IEnumerator<double> list = output.GetEnumerator(); 

            using (StreamWriter writer = new StreamWriter(stream))
            {
                while (list.MoveNext())
                {
                    writer.WriteLine(list.Current);
                }
            }
        }
    }
}