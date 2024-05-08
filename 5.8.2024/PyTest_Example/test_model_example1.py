import pytest
import pandas as pd
from model_example1 import preprocess_data, train_model, evaluate_model

@pytest.fixture
def test_data():
    data = {
        'PassengerId': [1310, 1311, 1312],
        'Survived': [1, 0, 0],
        'Pclass': [1, 3, 3],
        'Name': ['Name1310, Miss. Surname1310', 'Name1311, Col. Surname1311', 'Name1312, Mr. Surname1312'],
        'Sex': ['female', 'male', 'male'],
        'Age': [None, 29, 20],
        'SibSp': [0, 0, 0],
        'Parch': [0, 0, 0],
        'Ticket': ['SOTON/O2 3101272', '223596', '54636'],
        'Fare': [76.76016505, 10.19309671, 12.02941641],
        'Cabin': ['', '', 'C83'],
        'Embarked': ['C', 'S', 'C']
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
