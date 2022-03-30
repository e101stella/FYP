using MathNet.Numerics.LinearAlgebra;

namespace BCM
{
    class BCM_Multi
    {
        public static Matrix<double> Solver(Matrix<double> A, int updates)
        {
            // Does not converge properly!
            int n = A.RowCount;

            int rank = (int) Math.Sqrt(n);

            // Generating sigma matrix and normalising so that all rows equal 1.
            Matrix<double> sigma = CreateMatrix.Random<double>(n, rank).NormalizeRows(2D);

            // Creating gradient array and fililng with information
            Matrix<double> grad = (A - CreateMatrix.SparseOfDiagonalVector<double>(A.Diagonal())).Multiply(sigma);

            Vector<double> mag_grad = grad.RowNorms(2D);

            // Loop variables
            double[] grad_diff = new double[n];
            double[] max_val = new double [updates];
            int[]? ik = new int[updates], old_sel = new int [updates];

            Matrix<double> old_rows = CreateMatrix.Dense<double>(updates, rank);

            int max_iter = (int)Math.Pow(n, 2), iterations = 0;

            while (true)
            {
                // Resetting max values and index
                Parallel.For(0, n, i =>
                {
                    max_val[i] = -1;
                    ik[i] = -1;
                });

                // Calculating all gradients concurrently
                Parallel.For(0, n, i => {
                    grad_diff[i] = mag_grad[i] - sigma.Row(i).DotProduct(grad.Row(i));
                });

                for (int i = 0; i < n; i++)
                {   
                    // Finding first number that the current index is greater than
                    for (int j = 0; j < updates; j++)
                    {
                        if (grad_diff[i] > max_val[j])
                        {
                            // Shuffling values down
                            for (int k = updates - 1; k >= j; k--)
                            {
                                max_val[k] = max_val[k - 1];
                                ik[k] = ik[k - 1];
                            }

                            ik[j] = i;
                            max_val[j] = grad_diff[i];
                             
                            break;
                        }
                    }

                }

                // Returning if selection is same as old or small
                if (ik == old_sel || max_val[0] < 0.001 || iterations >= max_iter)
                {
                    return sigma;
                }
                old_sel = ik;

                // Updating sigma and recalculating gradient array.
                Parallel.For(0, updates, i => {
                    old_rows.SetRow(i, sigma.Row(ik[i]));
                    sigma.SetRow(ik[i], grad.Row(ik[i]) / mag_grad[ik[i]]);
                });

                // Updating gradient array
                for (int j = 0; j < updates; j++)
                {
                    int update = ik[j];
                    Parallel.For(0, n, i => {
                        if (i != update)
                        {
                            grad.SetRow(i, grad.Row(i) + A[i, update] * sigma.Row(update) - A[i, update] * old_rows.Row(update));
                        }
                    });
                }
                   

                mag_grad = grad.RowNorms(2D);
                iterations++;
            }

        }
    }
}