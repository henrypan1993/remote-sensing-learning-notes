import matplotlib.pyplot as plt
import numpy as np

# 设置画布
plt.figure(figsize=(10, 6))
plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文

# 1. 生成“振荡下降”的损失曲线（模拟学习率过大/梯度方向多变的情况）
x = np.arange(0, 50, 1)  # 迭代次数：0~49
# 基础下降趋势 + 正弦波动（模拟振荡）
loss_oscillate = 5 - 0.1*x + 0.8*np.sin(x*1.2)  # 波动幅度0.8，频率较高

# 2. 生成“平稳下降”的损失曲线（模拟合理学习率/加动量的情况）
loss_smooth = 5 - 0.1*x + 0.1*np.sin(x*0.5)  # 波动幅度0.1，频率低

# 绘图
plt.plot(x, loss_oscillate, color='#ff6b6b', linewidth=2.5, 
         marker='o', markersize=4, label='Oscillating Descent (e.g., High Learning Rate)')
plt.plot(x, loss_smooth, color='#4ecdc4', linewidth=2.5, 
         marker='s', markersize=4, label='Smooth Descent (e.g., Reasonable Learning Rate + Momentum)')

# 添加标注和标题
plt.xlabel('Iteration Count', fontsize=12)
plt.ylabel('Model Loss Value', fontsize=12)
plt.title('The oscillations in neural network training', fontsize=14, pad=20)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)  # 网格线（增强可读性）

# 突出标注“振荡区域”
plt.annotate('Oscillation Region: Loss values fluctuate violently up and down', 
             xy=(15, loss_oscillate[15]), xytext=(20, 3.5),
             arrowprops=dict(arrowstyle='->', color='#ff6b6b', alpha=0.7),
             fontsize=10, color='#ff6b6b', weight='bold')

plt.show()