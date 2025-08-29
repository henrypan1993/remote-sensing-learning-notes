import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子
np.random.seed(42)

# 模拟真实总体（总体方差已知）
population = np.random.normal(loc=0, scale=1, size=100000)  # 均值0, 方差1

# 多次抽样并计算方差（分母K vs K-1）
sample_size = 10
n_trials = 5000

var_with_K = []
var_with_K_minus_1 = []

for _ in range(n_trials):
    sample = np.random.choice(population, sample_size, replace=False)
    # 分母用K（有偏）
    var_K = np.sum((sample - np.mean(sample))**2) / sample_size
    # 分母用K-1（无偏）
    var_Km1 = np.sum((sample - np.mean(sample))**2) / (sample_size - 1)
    var_with_K.append(var_K)
    var_with_K_minus_1.append(var_Km1)

# 绘制直方图对比
plt.figure(figsize=(7, 5))
plt.hist(var_with_K, bins=40, alpha=0.6, label='Denominator = K (Biased)', color='red')
plt.hist(var_with_K_minus_1, bins=40, alpha=0.6, label='Denominator = K-1 (Unbiased)', color='blue')

# 添加真实方差参考线
plt.axvline(np.var(population), color='black', linestyle='dashed', linewidth=2, label='True variance')

plt.title('Biased vs Unbiased Variance Estimation', fontsize=14, fontweight='bold')
plt.xlabel('Estimated Variance')
plt.ylabel('Frequency')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
