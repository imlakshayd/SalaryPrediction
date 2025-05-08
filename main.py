import sys
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("TkAgg")  # This is used to show a plot in another window
import matplotlib.pyplot as plt

directory = "Data/Salary_dataset.csv"

pd.options.display.max_rows = 100
df = pd.read_csv(directory, index_col=False)

x = df['YearsExperience'].to_numpy()
y = df['Salary'].to_numpy()


def linear_regression(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    m = numerator / denominator
    b = y_mean - (m * x_mean)
    return m, b


def predict(m, x, b):
    return m * x + b


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


m, b = linear_regression(x, y)

predictions = predict(x, m, b)

error = rmse(y, predictions)

print("Slope (m):", m)
print("Intercept (b):", b)
print("Predictions:", predictions)
print("RMSE:", error)

print("--- DataFrame ---")
print(df.to_string(index=False))


plt.scatter(x, y, label="Data")
sorted_idx = np.argsort(x)
plt.plot(x[sorted_idx], predictions[sorted_idx], linewidth=2, label='Fit')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Salary vs. Years of Experience")
plt.grid(True)
plt.show()

