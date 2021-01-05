import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

df = pd.read_csv("prices.csv")
print(df.head())
print(df.info())

X = df[['lot_area', 'living_area', 'num_floors', 'num_bedrooms', 'num_bathrooms', 'waterfront', 'year_built', 'year_renovated']].values
Y = df['price'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

reg = LinearRegression()
reg.fit(X_train, Y_train)
prediction = reg.predict(X_test)

print("Score : ", reg.score(X_test, Y_test))
print("Root Mean Squared Error : ", np.sqrt(metrics.mean_squared_error(Y_test, prediction)))




