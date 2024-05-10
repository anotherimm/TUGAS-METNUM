# Nama		    : Imam Baihaqqy
# NIM		    : 21120122130078
# Mata Kuliah	: Metode Numerik
# Kelas		    : D

import numpy as np
import unittest

def inverse_matrix(matrix):
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return None

class TestInverseMatrix(unittest.TestCase):
    
    def test_inverse(self):
        matrix = np.array([[1, -1, 2], 
                           [3, 0, 1], 
                           [1, 0, 2]])
        expected_result = np.array([[0.0, 0.4, -0.2], 
                                    [-1.0, 0.0, 1.0], 
                                    [0.0, -0.2, 0.6]])
        self.assertTrue(np.allclose(inverse_matrix(matrix), expected_result))
        print ("Matrix: ")
        print(matrix)
        print ("Inverse matrix yang diharapkan:")
        print (expected_result)
        print ("Hasil Perhitungan:")
        print (inverse_matrix(matrix))
    def test_singular_matrix(self):
        matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.assertIsNone(inverse_matrix(matrix))

if __name__ == '__main__':
    unittest.main()
