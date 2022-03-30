using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.LinearAlgebra.Storage;
using MathNet.Numerics.LinearAlgebra;

class SparseIO {
    public static void Writer(string filename, Matrix<double> input){
        FileStream stream = new FileStream(filename, FileMode.Create);

        // Since it's a symmetric matrix only storing one of the triangles (potential for slight mem optimisation here in actual program.)
        SparseMatrix matrix = (SparseMatrix) input.LowerTriangle();

        // Location of the actual information about the matrix.
        SparseCompressedRowMatrixStorage<double> ms = (SparseCompressedRowMatrixStorage<double>) matrix.Storage;

        using (StreamWriter writer = new StreamWriter(stream)){
            // Writing number of non-zeros,
            writer.WriteLine($"{ms.RowCount} {ms.ColumnCount} {ms.ValueCount}");

            int curr_row = 0;

            for (int i = 0; i < matrix.NonZerosCount; i++){
                if (ms.RowPointers[curr_row+1] == i) {++curr_row;}
                writer.WriteLine($"{curr_row} {ms.ColumnIndices[i]} {ms.Values[i]}");
            }
        }
    } 

    public static Matrix<double> Reader(string filename){
        // Checking the file exists and openning it.
        FileStream stream = new FileStream(filename, FileMode.Open);

        using (StreamReader reader = new StreamReader(stream)){
            // Reading header and throwing exception with null file
            string? line = reader.ReadLine();
            if (line == null){throw new ArgumentNullException();}

            string[] values = line.Split(' ');
            int n = Convert.ToInt32(values[2]);
            
            Matrix<double> matrix = CreateMatrix.Sparse<double>(Convert.ToInt32(values[0]), Convert.ToInt32(values[1]));

            int row, col;
            double val;

            for (int i = 0; i < n; i++){
                line = reader.ReadLine();
                if (line == null){throw new ArgumentNullException();}

                values = line.Split(' ');
                row = Convert.ToInt32(values[0]);
                col = Convert.ToInt32(values[1]);

                val = Convert.ToDouble(values[2]);

                // Mirroring if not on diagonal   
                if (row != col){
                    matrix[col, row] = val;
                }

                matrix[row, col] = val;
            }
            return matrix;
        }
    }
}