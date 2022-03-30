using MathNet.Numerics.LinearAlgebra;

namespace BCM
{
    class BCM_AllAtOnce{
        public static double[] Solver(Matrix<double> A){
            // Does not converge properly!
            int n = A.RowCount;

            int rank = (int) Math.Sqrt(n); 

            // Generating sigma matrix and normalising so that all rows equal 1.
            Matrix<double> sigma = CreateMatrix.Random<double>(n, rank);
            Vector<double> sigNorm = sigma.RowNorms(2D);

            for (int i = 0; i < n; i++){
                sigma.SetRow(i, sigma.Row(i)/sigNorm[i]);
            }

            // Creating gradient array and fililng with information
            Matrix<double> grad = (A - CreateMatrix.SparseOfDiagonalVector<double>(A.Diagonal())).Multiply(sigma);

            Vector<double> mag_grad = grad.RowNorms(2D);

            double[] solutions = new double[n];
            
            for (int i = 0; i < n; i++){
                // Updating sigma and recalculating gradient array.
                sigma = grad.NormalizeRows(2D);
                grad = A * sigma;

                solutions[i] = Program.Solution(A, sigma);
            }
            return solutions;
        }
    }
}