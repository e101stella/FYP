using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra.Double;

namespace Generator{
    class Generator{
        public static void GenerateSymmetricMatrix(int number, int order, int degree, double mean, double std_dev) {
            /*
            Generates a number of sparse matrices of size order x order, with a density of degree/order. 
            Populates values according to a gaussian distribution with associated variables mean & std_dev
            Saves the generated matrices to binary files
            */
            Matrix<double> matrix;

            // Instantiating random number generator and normal distribution
            Random source = new Random();
            Normal dist = new Normal(mean, std_dev, source);

            int col;
            string filename;

            for (int i = 1; i <= number; i++){
                // Creating new empty matrix
                matrix = CreateMatrix.Sparse<double>((int) order, order);

                // Iterating over the rows
                for (int row = 0; row < order; row++){
                    // Sampling the standard normal distribution for the diagonal
                    matrix[row, row] = dist.Sample();

                    for (int num = 0; num < degree; num++){
                        // Getting the column to assign and making sure it does not correspond to the diagonal.
                        // Casting to an int since it outputs a long.
                        col = (int) source.NextInt64(order);
                        
                        while (col == row){
                            col = (int) source.NextInt64(order);
                        }

                        matrix[row, col] = dist.Sample();
                    }
                }
                filename = $"TestCases/{i}n{order}_d{degree}_m{mean}_std{std_dev}.csv";

                matrix = 0.5 * (matrix + matrix.Transpose());
                SparseIO.Writer(filename, matrix);
            }
        }
    }
}