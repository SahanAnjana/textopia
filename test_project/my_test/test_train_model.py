import unittest
import numpy as np
from model import train_model

class TestModelTraining(unittest.TestCase):
    def test_model_accuracy(self):
        X_train = np.random.rand(100, 10)
        y_train = np.random.randint(2, size=100)

        X_test = np.random.rand(20, 10)
        y_test = np.random.randint(2, size=20)
        
        accuracy = train_model(X_train, y_train,X_test, y_test)

        self.assertTrue(accuracy >= 0 and accuracy <= 1, "Accuracy out of range.")

if __name__ == '__main__':
    unittest.main()
