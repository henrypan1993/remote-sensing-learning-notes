import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 构造一个非高斯分布：双峰分布 (两个正态分布的混合)
x = np.linspace(-6, 6, 500)
true_dist = 0.6 * norm.pdf(x, loc=-2, scale=1) + 0.4 * norm.pdf(x, loc=2, scale=0.8)

# 单高斯近似 (均值=0, 方差=2)
approx_single = norm.pdf(x, loc=0, scale=2)

# 多高斯拟合 (用两个高斯逼近真实分布)
approx_mixture = 0.5 * norm.pdf(x, loc=-2, scale=1) + 0.5 * norm.pdf(x, loc=2, scale=1)

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(x, true_dist, label="True Distribution (Non-Gaussian)", color="black", linewidth=2)
plt.plot(x, approx_single, label="Single Gaussian Approximation", color="red", linestyle="--")
plt.plot(x, approx_mixture, label="Mixture of Gaussians Approximation", color="blue", linestyle=":")

plt.title("True Distribution vs Gaussian Approximations")
plt.xlabel("x")
plt.ylabel("Probability Density")
plt.legend()
plt.grid(True)
plt.show()
