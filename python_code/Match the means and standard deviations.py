import numpy as np
import matplotlib.pyplot as plt

# 生成模拟数据：两个探测器的信号分布
np.random.seed(42)
# 探测器A：均值100，标准差20
signal_A = np.random.normal(100, 20, 1000)
# 探测器B：均值140，标准差35
signal_B = np.random.normal(140, 35, 1000)

# 校正：匹配B到A的均值和标准差
mean_A, std_A = np.mean(signal_A), np.std(signal_A)
mean_B, std_B = np.mean(signal_B), np.std(signal_B)

# Z-score标准化 + 平移
signal_B_corrected = (signal_B - mean_B) * (std_A / std_B) + mean_A

# 绘图
plt.figure(figsize=(10,5))

# 校正前直方图
plt.subplot(1, 2, 1)
plt.hist(signal_A, bins=30, alpha=0.6, label='Detector A')
plt.hist(signal_B, bins=30, alpha=0.6, label='Detector B')
plt.title("Before Matching")
plt.xlabel("Signal Value")
plt.ylabel("Frequency")
plt.legend()

# 校正后直方图
plt.subplot(1, 2, 2)
plt.hist(signal_A, bins=30, alpha=0.6, label='Detector A')
plt.hist(signal_B_corrected, bins=30, alpha=0.6, label='Detector B (corrected)')
plt.title("After Matching Means & Std Devs")
plt.xlabel("Signal Value")
plt.ylabel("Frequency")
plt.legend()

plt.tight_layout()
plt.show()
