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


def predict(x, m, b):
    return m * x + b


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def cost(m, b, x, y):
    n = len(x)
    cost = 1/(2*n)*(np.sum((m * x + b - y)**2))
    return cost

def partiald_m(m, b, x, y):
    n = len(x)
    partiald_m = (1/n) * (np.sum((m * x + b - y) * x))
    return partiald_m

def partiald_b(m, b, x, y):
    n = len(x)
    partiald_b = (1/n) * (np.sum(m * x + b -y))
    return partiald_b

def descent(alpha, m, b, x, y, iterations):
    cost_history = []
    i = 0
    while i < iterations:
        m = m - alpha * partiald_m(m, b, x, y)
        b = b - alpha * partiald_b(m, b, x, y)
        i += 1
        cost_history.append(cost(m, b, x, y))
    return m, b, cost_history

m, b = 0.0, 0.0
alpha = 0.005
iterations = 10000

m_gd, b_gd = descent(alpha, m, b, x, y, iterations)

m_cf, b_cf = linear_regression(x, y)

y_pred_cf = predict(x, m_cf, b_cf)
y_pred_gd = predict(x, m_gd, b_gd)

error_cf = rmse(y, y_pred_cf)
error_gd = rmse(y, y_pred_gd)

print(f"GD slope: {m_gd:.2f}, GD intercept: {b_gd:.2f}")
print(f"CF slope: {m_cf:.2f}, CF intercept: {b_cf:.2f}")
print("Predictions:", y_pred_cf)
print("Closed-form RMSE:", error_cf)
print("Gradient Descent RMSE:", error_gd)


print("--- DataFrame ---")
print(df.to_string(index=False))


plt.scatter(x, y, label="Data")
sorted_idx = np.argsort(x)
plt.plot(x[sorted_idx], y_pred_cf[sorted_idx], linewidth=2, label='Fit')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Salary vs. Years of Experience")
plt.grid(True)
plt.show()

# for α in [0.0005, 0.001, 0.005]:
#     m_gd, b_gd, costs = descent(α, 0, 0, x, y, 1000)
#     plt.plot(costs, label=f'α={α}')
# plt.legend(); plt.show()
