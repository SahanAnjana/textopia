import unittest
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
from model_example1 import load_data, preprocess_data, build_model, train_model, test_model, save_model

class TestLoanModel(unittest.TestCase):

    def setUp(self):
        self.data = pd.DataFrame({
            'TARGET': np.random.randint(0, 2, 100),
            'SK_ID_CURR': np.random.randint(100000, 200000, 100),
            'AMT_INCOME_TOTAL': np.random.randint(20000, 200000, 100),
            'NAME_CONTRACT_TYPE': ['Cash loans' if i%2==0 else 'Revolving loans' for i in range(100)]
        })

    def test_load_data(self):
        data = load_data('./application_full.csv')
        self.assertIsInstance(data, pd.DataFrame),"Data is mismatched"

    def test_preprocess_data(self):
        X_train, X_test, y_train, y_test = preprocess_data(self.data)
        self.assertEqual(X_train.shape[0], 80),"X_train shape is mismatched"
        self.assertEqual(X_test.shape[0], 20),"X_test shape is mismatched"
        self.assertEqual(len(y_train), 80),"y_train shape is mismatched"
        self.assertEqual(len(y_test), 20),"y_test shape is mismatched"

    def test_build_model(self):
        model = build_model()
        self.assertIsInstance(model, DecisionTreeClassifier),"Model is not DecisionTreeClassifier"

    def test_train_model(self):
        model = build_model()
        X_train, X_test, y_train, y_test = preprocess_data(self.data)
        train_model(model, X_train, y_train)
        self.assertTrue(hasattr(model, 'tree_')),"Model is not trainable"

    def test_test_model(self):
        model = build_model()
        X_train, X_test, y_train, y_test = preprocess_data(self.data)
        train_model(model, X_train, y_train)
        accuracy = test_model(model, X_test, y_test)
        self.assertIsInstance(accuracy, float)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_save_model(self):
        model = DecisionTreeClassifier()
        save_model(model, 'test_model.pkl')
        self.assertTrue(os.path.exists('test_model.pkl'))

    def tearDown(self):
        if os.path.exists('test_model.pkl'):
            os.remove('test_model.pkl')


if __name__ == '__main__':
    unittest.main()
