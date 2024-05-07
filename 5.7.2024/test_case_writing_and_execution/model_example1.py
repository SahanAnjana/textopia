import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib


def load_data(path):
    data = pd.read_csv(path)
    return data

def preprocess_data(data):
    data.dropna(thresh=len(data) * 0.9, axis=1, inplace=True)
    X = data.drop(['TARGET', 'SK_ID_CURR'], axis=1)
    y = data['TARGET']

    X_new=build_preprocessing_pipeline(X)

    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def build_preprocessing_pipeline(X):
    categorical_features = X.select_dtypes(include=['object']).columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # pipeline = Pipeline(steps=[
    #     ('preprocessor', preprocessor),
    #     ('pca', PCA(n_components=70))
    # ])

    X_new = preprocessor.fit_transform(X)

    return X_new

def build_model():
    model= DecisionTreeClassifier()
    return model

def train_model(model,X_train,y_train):
    model.fit(X_train, y_train)

def test_model(model,X_test,y_test):
    y_pred = model.predict(X_test)
    threshold = 0.5
    y_pred_binary = (y_pred >= threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred_binary)
    return accuracy

def save_model(model, filepath):
    joblib.dump(model, filepath)


if __name__=="__main__":
    data=load_data('./application_full.csv')
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = build_model()
    train_model(model, X_train, y_train)
    accuracy = test_model(model, X_test, y_test)
    print("Accuracy:", accuracy)
    save_model(model,'logreg_model.pkl')
    print("Model saved successfully!!!")

