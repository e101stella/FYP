using System.IO;
using MathNet.Numerics.LinearAlgebra;

namespace BCM{
    class BCM{
        public static Matrix<double> Solver(Matrix<double> A){
            int n = A.RowCount;

            int rank = (int) Math.Sqrt(n); // (int) Math.Min(100D, Math.Sqrt(2*n));

            // Generating sigma matrix and normalising so that all rows equal 1.
            Matrix<double> sigma = CreateMatrix.Random<double>(n, rank).NormalizeRows(2D);

            // Creating gradient array and fililng with information
            Matrix<double> noDiag = A - CreateMatrix.SparseOfDiagonalVector<double>(A.Diagonal());
            Matrix<double> grad = (A - CreateMatrix.SparseOfDiagonalVector<double>(A.Diagonal())).Multiply(sigma);

            Vector<double> mag_grad = grad.RowNorms(2D);

            // Declaring loop variables
            double max_val, trace;
            int ik, old_sel=-1;
            
            while (true){
                // Max gradient selection
                max_val = mag_grad[0] - sigma.Row(0).PointwiseMultiply(grad.Row(0)).Sum();
                ik = 0;

                for (int i = 1; i < n; i++){
                    trace = sigma.Row(i).DotProduct(grad.Row(i));
                    if (mag_grad[i] - trace > max_val){
                        max_val = mag_grad[i] - trace;
                        ik = i;
                    }
                }       

                // Returning if selection is same as old or small
                if (ik == old_sel || max_val < 0.001) {
                    return sigma;
                    }
                old_sel = ik;

                // Updating sigma and recalculating gradient array.
                sigma.SetRow(ik, grad.Row(ik)/mag_grad[ik]);
                grad = (A - CreateMatrix.SparseOfDiagonalVector<double>(A.Diagonal())) * sigma;

                mag_grad = grad.RowNorms(2D);
            } 
        }
    }
}