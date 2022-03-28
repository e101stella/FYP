using MathNet.Numerics.LinearAlgebra;

class Program{
    public static void Main(string[] args){
        // Generator.Generator.GenerateNewMatrix(1, 10000, 5, 0, 1); 

        Matrix<double> input = SparseIO.Reader("TestCases/1n1000_d5_m0_std1.csv");
        Matrix<double> output = ThreadedBCM.BCM.Solver(-input);

        double opt_sol = input.PointwiseMultiply(output * output.Transpose()).RowSums().Sum(); 
        Console.Write(opt_sol);
        return;
    }
}