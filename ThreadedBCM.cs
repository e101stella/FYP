using System;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics;

namespace BCM{
    class ThreadedBCM{
        public static Matrix<double> Solver(Matrix<double> A){
            Control.MaxDegreeOfParallelism = 24;
            int n = A.RowCount;

            int rank = (int) Math.Sqrt(n);

            // Generating sigma matrix and normalising so that all rows equal 1.
            Matrix<double> sigma = CreateMatrix.Random<double>(n, rank);
            Vector<double> sigNorm = sigma.RowNorms(2D);

            for (int i = 0; i < n; i++){
                sigma.SetRow(i, sigma.Row(i)/sigNorm[i]);
            }

            // Creating gradient array and fililng with information
            Matrix<double> noDiag = A - CreateMatrix.SparseOfDiagonalVector<double>(A.Diagonal());
            Matrix<double> grad = (A - CreateMatrix.SparseOfDiagonalVector<double>(A.Diagonal())).Multiply(sigma);

            Vector<double> mag_grad = grad.RowNorms(2D);

            // Declaring loop variables
            double[] grad_diff = new double[n];
            double max_val;
            int ik, old_sel=-1;

            int max_iter = (int) Math.Pow(n, 2), iterations = 0;
            
            while (true){
                // Calculating all gradients concurrently
                Control.UseSingleThread();
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
                    return sigma;
                    }
                old_sel = ik;
                
                Control.UseMultiThreading();
                // Updating sigma and recalculating gradient array.
                sigma.SetRow(ik, grad.Row(ik)/mag_grad[ik]);
                grad = (A - CreateMatrix.SparseOfDiagonalVector<double>(A.Diagonal())) * sigma;

                mag_grad = grad.RowNorms(2D);
                iterations++;
            } 
        }
    }
}