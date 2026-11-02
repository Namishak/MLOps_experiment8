
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def test_clean_data():
    X = pd.DataFrame({"A": [1, np.nan, 3]})
    result = X.fillna(X.mean())
    assert result.isnull().sum().sum() == 0

def test_data_split():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    assert len(X_train) > 0
    assert len(X_test) > 0

def test_model_training():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)
    assert model is not None
    assert model.coef_.shape[1] == data.data.shape[1]

def test_model_prediction():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    assert len(predictions) == len(y_test)
    assert set(predictions).issubset({0, 1})
