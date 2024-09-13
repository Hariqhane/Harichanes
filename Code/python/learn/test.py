import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.font_manager as fm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']

# Provided data
x = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.42, 0.44, 0.46, 0.48, 0.5, 0.52, 0.54, 0.56, 0.58, 0.6, 0.7, 0.8, 0.9])
y = np.array([0, 0.027, 0.088, 0.253, 44.4, 54.8, 78.7, 95, 111.5, 126.5, 144.8, 158.9, 181.4, 199.6, 215.4, 301.2, 398.3, 471.3])

# Define the type of function we expect the data to follow
def model_func(x, a, b, c):
    return a * x**2 + b * x + c

# Fit the model function to the data
popt, pcov = curve_fit(model_func, x, y)

# Generate a smooth line for plotting the fit
x_fit = np.linspace(0, 0.9, 100)
y_fit = model_func(x_fit, *popt)
# Calculate the R-squared value for the fit
residuals = y - model_func(x, *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y - np.mean(y))**2)
r_squared = 1 - (ss_res / ss_tot)
r_squared
# Plot the data and the fit
# Modify the plot with Chinese annotations and titles
plt.figure(figsize=(10, 5))
plt.scatter(x, y, label='数据点')
plt.plot(x_fit, y_fit, 'r-', label=f'拟合曲线')
plt.title('P-I拟合曲线')
plt.xlabel('电流/A')
plt.ylabel('功率P/mW')

plt.legend()

plt.show()


# Output the fit coefficients
popt, np.sqrt(np.diag(pcov))
