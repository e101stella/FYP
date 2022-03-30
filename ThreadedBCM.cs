using System;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics;

namespace BCM{
    class ThreadedBCM{
        public static IEnumerable<double> Solver(Matrix<double> A){
            Control.MaxDegreeOfParallelism = 24;
            int n = A.RowCount;

            int rank = (int) Math.Sqrt(n);

            IEnumerable<double> output = Enumerable.Empty<double>();

            // Generating sigma matrix and normalising so that all rows equal 1.
            Matrix<double> sigma = CreateMatrix.Random<double>(n, rank).NormalizeRows(2D);

            // Creating gradient array and fililng with information
            Matrix<double> noDiag = A - CreateMatrix.SparseOfDiagonalVector<double>(A.Diagonal());
            Matrix<double> grad = (A - CreateMatrix.SparseOfDiagonalVector<double>(A.Diagonal())).Multiply(sigma);

            Vector<double> mag_grad = grad.RowNorms(2D);

            // Since using explicit multi-threading disabling built-in multithreading
            Control.UseSingleThread();

            // Declaring loop variables
            double[] grad_diff = new double[n];
            double max_val;
            int ik, old_sel=-1;
            Vector<double> old_row;

            int max_iter = (int) Math.Pow(n, 2), iterations = 0;
            
            while (true){
                // Calculating all gradients concurrently
                Parallel.For(0, n, i => {
                    grad_diff[i] = mag_grad[i] - sigma.Row(i).DotProduct(grad.Row(i));
                });
                
                // Finding the maxima natively
                max_val = -1;
                ik = -1;
                for (int i = 0; i < n; i++){
                    if (grad_diff[i] > max_val){
                        ik = i;
                        max_val = grad_diff[i];
                    }
                }

                // Returning if selection is same as old or small
                if (ik == old_sel || max_val < 0.001 || iterations >= max_iter) {
                    return output;
                    }
                old_sel = ik;

                // Updating sigma and recalculating gradient array.
                old_row = sigma.Row(ik);
                sigma.SetRow(ik, grad.Row(ik)/mag_grad[ik]);

                // Updating gradient array
                Parallel.For(0, n, i => {
                    if (i != ik){
                        grad.SetRow(i, grad.Row(i) + A[i,ik] * sigma.Row(ik) - A[i, ik] * old_row);
                    }
                });

                mag_grad = grad.RowNorms(2D);
                iterations++;
                output = output.Append(Program.Solution(A, sigma));
            } 
        }
    }
}