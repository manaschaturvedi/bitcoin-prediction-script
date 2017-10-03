import quandl, math
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

df = pd.read_csv('bitcoin_archive_data.csv')
df = df[['Price','Open','High','Low','Change %']]
df['HL_PCT'] = ((df['High']-df['Low'])/(df['Low']))*100.0
df = df[['Price','HL_PCT','Change %']]

forecast_col = 'Price'
df.fillna(value=-99999, inplace=True)
# forecast_out = int(math.ceil(0.01 * len(df)))
forecast_out = 7
df['label'] = df[forecast_col].shift(forecast_out)

# print df

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[:forecast_out]
X = X[forecast_out:]

df.dropna(inplace=True)

y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print(confidence)
forecast_set = clf.predict(X_lately)
print forecast_set