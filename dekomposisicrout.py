# Nama		    : Imam Baihaqqy
# NIM		    : 21120122130078
# Mata Kuliah	: Metode Numerik
# Kelas		    : D

import numpy as np
import unittest
def crout(A):
    if A.shape[0] != A.shape[1]:
        raise ValueError('A must be a square matrix')

    n = A.shape[0]
    L = np.zeros((n, n))
    U = np.eye(n)

    for i in range(n):
        L[i, 0] = A[i, 0]

    for j in range(1, n):
        U[0, j] = A[0, j] / L[0, 0]

    for i in range(1, n):
        for j in range(1, i + 1):
            L[i, j] = A[i, j] - np.dot(L[i, 0:j], U[0:j, j])

        for j in range(i + 1, n):
            U[i, j] = (A[i, j] - np.dot(L[i, 0:i], U[0:i, j])) / L[i, i]

    return L, U

# Test the crout function
A = np.array([[2, 4, 3],
              [3, 5, 2],
              [4, 6, 3]])
L, U = crout(A)
print("Lower triangular matrix L:")
print(L)
print("\nUpper triangular matrix U:")
print(U)

class TestCrout(unittest.TestCase):

    def test_decomposition(self):
        L, U = crout(A)
        reconstructed_A = np.dot(L, U)
        np.testing.assert_allclose(reconstructed_A, A, atol=1e-10)

    def test_non_square_matrix(self):
        A = np.array([[1, 2, 3], [4, 5, 6]])  # Non-square matrix
        with self.assertRaises(ValueError):
            crout(A)

if __name__ == '__main__':
    unittest.main()