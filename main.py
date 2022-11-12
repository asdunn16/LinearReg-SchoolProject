import pandas as pd
import numpy as np
# from sklearn.datasets import load_boston
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# housing_data = load_boston()
housing_data = fetch_california_housing()
housing_data_df = pd.DataFrame(housing_data.data, columns=housing_data.feature_names)
housing_data_df['Price'] = housing_data.target

print(housing_data_df.head())
print(housing_data_df.describe())

X = housing_data.data
y = housing_data.target
print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictions))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, predictions))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

