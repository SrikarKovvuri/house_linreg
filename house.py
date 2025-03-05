import numpy as numpy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn
import matplotlib.pyplot as plt

df = pd.read_csv('house_prices.csv')

x = df[['Square Footage']]
y = df['Price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=45)

model = LinearRegression()

model.fit(x_train, y_train)

y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print("completed")
plt.scatter(x_test, y_test, color='blue', label="Actual Prices")
plt.plot(x_test, y_pred, color='red', linewidth=2, label="Predicted Line")
plt.xlabel("Square Footage")
plt.ylabel("Price")
plt.legend()
plt.show()