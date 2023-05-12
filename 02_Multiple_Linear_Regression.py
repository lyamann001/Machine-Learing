
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score

df = pd.read_csv("data/FuelConsumption.csv")
print(df.head())

# Simple Linear Regression ile Multiple Linear Regression arasındaki fark, Simple'da bir tane bağımsız değişken varken Multiple'da birden fazla bağımsız değişkenimiz mevcut.

cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]

msk = np.random.randn(len(df)) <= 0.8
train = cdf[msk]
test = cdf[~msk]

regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(x, y)
print(f'Coefficiesnts: {regr.coef_}')

y_result = regr.predict(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
print(y_result)

print('Variance Socore: %.2f' % regr.score(x, y))