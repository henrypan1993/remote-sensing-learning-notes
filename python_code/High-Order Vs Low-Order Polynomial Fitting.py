import numpy as np
import matplotlib.pyplot as plt

# 数据
np.random.seed(0)
x = np.linspace(-3, 3, 10)
y = np.sin(x) + np.random.normal(scale=0.1, size=len(x))

# 多项式拟合
coeff_low = np.polyfit(x, y, 1)  # 一阶
coeff_high = np.polyfit(x, y, 7) # 七阶

# 拟合曲线
x_fit = np.linspace(-5, 5, 200)
y_low = np.polyval(coeff_low, x_fit)
y_high = np.polyval(coeff_high, x_fit)

# 绘图
plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='black', label='Control points (GCPs)')
plt.plot(x_fit, y_low, 'b-', label='Low-order polynomial (1st order)')
plt.plot(x_fit, y_high, 'r-', label='High-order polynomial (7th order)')

# 外推区域标记
plt.axvspan(-5, x.min(), color='gray', alpha=0.1)
plt.axvspan(x.max(), 5, color='gray', alpha=0.1)
plt.text(-4.8, 1.2, 'Extrapolation\nregion', fontsize=9)
plt.text(3.2, 1.2, 'Extrapolation\nregion', fontsize=9)

plt.title("High-order vs Low-order Polynomial Fitting")
plt.xlabel("Coordinate")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()
