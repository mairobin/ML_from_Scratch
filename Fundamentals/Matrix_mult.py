def matrix_vector_multiply_verbose(matrix, vector):
    # Get the dimensions of the matrix and vector
    rows_matrix, cols_matrix = len(matrix), len(matrix[0])
    size_vector = len(vector)

    # Check if the matrix and vector are compatible for multiplication
    if cols_matrix != size_vector:
        raise ValueError("Incompatible matrix and vector sizes for multiplication")

    # Initialize the result vector to zeros
    result = [0] * rows_matrix

    print("Matrix:\n", matrix)
    print("Vector:\n", vector)

    # Perform the matrix-vector multiplication with print statements
    for i in range(rows_matrix):
        for j in range(cols_matrix):
            print(f"Step {i + 1}-{j + 1}: {result[i]} + ({matrix[i][j]} * {vector[j]})")
            result[i] += matrix[i][j] * vector[j]

    print("\nResulting Vector:\n", result)

# Example usage
matrix = [[1, 2, 3],
          [4, 5, 6]]

vector = [7, 8, 9]

matrix_vector_multiply_verbose(matrix, vector)
