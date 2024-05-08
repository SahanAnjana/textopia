import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

def preprocess_data(df):
    df['Cabin'] = df['Cabin'].fillna("No cabin")
    df['Embarked'] = df['Embarked'].fillna("O")
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    mean_age = df['Age'].mean()
    df['Age'] = df['Age'].fillna(mean_age)
    
    df = df.drop(['Name'], axis=1)
    
    encoder = OneHotEncoder(sparse_output=True)
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    cat_df = df[categorical_columns]
    one_hot_encoded = encoder.fit_transform(cat_df)
    one_hot_df = pd.DataFrame.sparse.from_spmatrix(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))
    
    df = pd.concat([df, one_hot_df], axis=1)
    df = df.drop(categorical_columns, axis=1)
    
    return df

def train_model(df):
    df = preprocess_data(df)
    
    X = df.drop(['Survived'], axis=1)
    y = df[['Survived']]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2, test_size=0.3)
    
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    report = classification_report(y_test, y_pred)
    
    return accuracy, report

if __name__ == "__main__":
    df = pd.read_csv('huge_1M_titanic.csv')
    
    trained_model, X_test, y_test = train_model(df)
    
    accuracy, report = evaluate_model(trained_model, X_test, y_test)

    print("Accuracy:", accuracy)
    print("Classification Report:\n", report)
