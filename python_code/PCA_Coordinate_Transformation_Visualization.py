import matplotlib.pyplot as plt
import numpy as np

# 设置随机数种子
np.random.seed(0)

# 原始协方差矩阵
mean = [0, 0]
cov = [[3, 2], [2, 2]]
x = np.random.multivariate_normal(mean, cov, 500)

# PCA旋转
eigvals, eigvecs = np.linalg.eigh(cov)

# 作图：三合一
fig, ax = plt.subplots(figsize=(7,7))

# 绘制点云
ax.scatter(x[:,0], x[:,1], alpha=0.3, s=10)

# 原始坐标轴 (x1, x2)
ax.arrow(0, 0, 3, 0, head_width=0.2, head_length=0.3, fc='blue', ec='blue', linewidth=2)
ax.text(3.4, -0.2, "x1", color='blue', fontsize=12, ha='center')
ax.arrow(0, 0, 0, 3, head_width=0.2, head_length=0.3, fc='blue', ec='blue', linewidth=2)
ax.text(-0.2, 3.4, "x2", color='blue', fontsize=12, ha='center')

# PCA坐标轴 (y1, y2)
scale = 3  # 伸缩因子，用于显示箭头
ax.arrow(0, 0, eigvecs[0,1]*scale, eigvecs[1,1]*scale, head_width=0.2, head_length=0.3, fc='red', ec='red', linewidth=2)
ax.text(eigvecs[0,1]*scale*1.3, eigvecs[1,1]*scale*1.3, "y1 (PC1)", color='red', fontsize=12, ha='center')
ax.arrow(0, 0, eigvecs[0,0]*scale, eigvecs[1,0]*scale, head_width=0.2, head_length=0.3, fc='red', ec='red', linewidth=2)
ax.text(eigvecs[0,0]*scale*1.3, eigvecs[1,0]*scale*1.3, "y2 (PC2)", color='red', fontsize=12, ha='center')

# 设置属性
ax.set_title("Coordinate System Transformation:\nOriginal Feature Space vs Principal Component Space", fontsize=14, pad=20)
ax.axhline(0, color='k', linewidth=0.5)
ax.axvline(0, color='k', linewidth=0.5)
ax.axis('equal')
ax.set_xlim(-4.5, 4.5)
ax.set_ylim(-4.5, 4.5)
ax.grid(True, alpha=0.3)

plt.show()
