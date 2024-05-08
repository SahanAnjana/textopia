import pytest
import pandas as pd
from model_example1 import preprocess_data, train_model, evaluate_model

@pytest.fixture
def test_data():
    data = {
        'PassengerId': [1, 2, 3],
        'Pclass': [1, 2, 3],
        'Sex': ['male', 'female', 'male'],
        'Age': [22, 35, None],
        'SibSp': [1, 0, 1],
        'Parch': [0, 0, 1],
        'Fare': [7.25, 71.2833, 8.05],
        'Cabin': ['C85', 'C85', None],
        'Embarked': ['S', 'C', 'S'],
        'Survived': [0, 1, 1]
    }
    return pd.DataFrame(data)

def test_preprocess_data(test_data):
    processed_data = preprocess_data(test_data)
    assert len(processed_data) == 3
    assert 'Cabin' not in processed_data.columns
    assert 'Embarked' not in processed_data.columns
    assert 'Sex' not in processed_data.columns
    assert 'male' in processed_data.columns
    assert 'female' in processed_data.columns
    assert processed_data['Age'].isnull().sum() == 0

def test_train_model(test_data):
    model, X_test, y_test = train_model(test_data)
    assert model is not None
    assert len(X_test) > 0
    assert len(y_test) > 0

def test_evaluate_model(test_data):
    model, X_test, y_test = train_model(test_data)
    accuracy, report = evaluate_model(model, X_test, y_test)
    assert accuracy >= 0.0 and accuracy <= 1.0
    assert isinstance(report, str)

if __name__ == "__main__":
    pytest.main()
