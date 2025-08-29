import numpy as np
import matplotlib.pyplot as plt

# 定义 Keys cubic convolution kernel
def cubic_kernel(x, a=-0.5):
    abs_x = np.abs(x)
    if abs_x <= 1:
        return (a + 2) * abs_x**3 - (a + 3) * abs_x**2 + 1
    elif abs_x < 2:
        return a * abs_x**3 - 5*a * abs_x**2 + 8*a * abs_x - 4*a
    else:
        return 0

# 生成数据
x_vals = np.linspace(-2.5, 2.5, 500)
y_vals = [cubic_kernel(x) for x in x_vals]

# 绘图
plt.figure(figsize=(6,4))
plt.plot(x_vals, y_vals, label="Cubic Convolution Kernel (a=-0.5)", color="blue")
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)
plt.axvline(1, color="gray", linestyle="--", linewidth=0.8)
plt.axvline(-1, color="gray", linestyle="--", linewidth=0.8)
plt.axvline(2, color="gray", linestyle="--", linewidth=0.8)
plt.axvline(-2, color="gray", linestyle="--", linewidth=0.8)
plt.title("Cubic Convolution Kernel Shape")
plt.xlabel("Distance from Target Pixel (in pixels)")
plt.ylabel("Weight")
plt.legend()
plt.grid(True)
plt.show()
