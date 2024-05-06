import unittest
import numpy as np
from model import preprocess_data


class TestDataProcessing(unittest.TestCase):
    def test_data_shape(self):
        # raw_data=np.random.rand(100,10)
        raw_data = np.random.randint(0, 100, size=(100, 10))
        processed_data=preprocess_data(raw_data)
        self.assertEqual(processed_data.shape,(100,10),"Data Shape Mismatch.")

    def test_data_range(self):
        # raw_data=np.random.rand(100,10)
        raw_data = np.random.randint(0, 100, size=(100, 10))
        processed_data=preprocess_data(raw_data)
        self.assertTrue(np.all(processed_data>=0) and np.all(processed_data<=1),"Data values are out of range.")


if __name__=="__main__":
    unittest.main()
