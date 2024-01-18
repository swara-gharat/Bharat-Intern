import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn import metrics
import seaborn as sns
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


df = pd.read_csv("D:/SWARA/internship/Housing.csv")
print(df)
df.describe()
df.info()
df.head(10)
print(df.isnull().sum())

sns.pairplot(df)
plt.show()

x = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]

y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()
model.add(Dense(8, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=1)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")


num_samples = 5
sample_indices = np.random.choice(len(y_test), num_samples, replace=False)
for idx in sample_indices:
    actual_price = y_test.iloc[idx]
    predicted_price = y_pred[idx][0]
    print(f"Actual Price: {actual_price:.2f}, Predicted Price: {predicted_price:.2f}")

#print(y_pred)
#print(y_test)

lm = LinearRegression()
lm.fit(X_train,y_train)

print(lm.intercept_)
coeff_df = pd.DataFrame(lm.coef_,x.columns,columns=['Coefficient'])
print(coeff_df)

predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)
plt.show()

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


num_samples = 5
sample_indices = np.random.choice(len(y_test), num_samples, replace=False)
for idx in sample_indices:
    actual_price = y_test.iloc[idx]
    predicted_price = y_pred[idx][0]
    print(f"Actual Price: {actual_price:.2f}, Predicted Price: {predicted_price:.2f}")



