import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veri = pd.read_csv('sp500.csv')

veri.drop(['date'], axis = 1, inplace = True)


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
name = veri.iloc[:,5:6].values
name = le.fit_transform(name)

veri.drop(['Name'], axis = 1, inplace = True)
name = pd.DataFrame(data = name, index = range(619040), columns = ['Name'])

tam_veri = pd.concat([veri, name], axis = 1)

kapanıs = tam_veri.iloc[:,3:4]
tam_veri.drop(['close'], axis = 1, inplace = True)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(tam_veri, kapanıs, test_size = 0.33, random_state = 0)



from sklearn.linear_model import LinearRegression


x_train = x_train.fillna(x_train.mean())
y_train = y_train.fillna(y_train.mean())
reg = LinearRegression()
reg.fit(x_train, y_train)
x_test = x_test.fillna(x_test.mean())
y_pred = reg.predict(x_test)
y_pred = pd.DataFrame(data = y_pred, index = range(204284), columns = ['y_pred'])
plt.plot(y_test, y_pred, 'blue')
plt.show()
