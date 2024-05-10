# Nama		    : Imam Baihaqqy
# NIM		    : 21120122130078
# Mata Kuliah	: Metode Numerik
# Kelas		    : D

import numpy as np
import unittest

def lu_decomposition(A):
    # Get the size of the matrix
    n = A.shape[0]

    # Initialize L and U matrices
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    # Perform Gaussian elimination
    for i in range(n):
        # Set diagonal elements of L to 1
        L[i][i] = 1

        # Compute upper triangular matrix U
        for k in range(i, n):
            total = sum(L[i][p] * U[p][k] for p in range(i))
            U[i][k] = A[i][k] - total

        # Compute lower triangular matrix L
        for k in range(i+1, n):
            total = sum(L[k][p] * U[p][i] for p in range(i))
            L[k][i] = (A[k][i] - total) / U[i][i]

    return L, U

# Define a sample matrix
A = np.array([[4, 3, -1], [-2, -4, 5], [1, 2, 6]])

# Perform LU decomposition
L, U = lu_decomposition(A)

print("Lower triangular matrix L:")
print(L)
print("\nUpper triangular matrix U:")
print(U)

class TestLUDecomposition(unittest.TestCase):
    
    def test_decomposition(self):
        A = np.array([[4, 3, -1], [-2, -4, 5], [1, 2, 6]])
        expected_L = np.array([[1., 0., 0.], [-0.5, 1., 0.], [0.25, -0.5, 1.]])
        expected_U = np.array([[4., 3., -1.], [0., -2.5, 4.5], [0., 0., 8.5]])
        L, U = lu_decomposition(A)
        self.assertTrue(np.allclose(L, expected_L))
        self.assertTrue(np.allclose(U, expected_U))

if __name__ == '__main__':
    unittest.main()