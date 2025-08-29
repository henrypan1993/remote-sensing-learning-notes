import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# 随机生成2D点簇
np.random.seed(42)
points = np.random.normal(loc=[2, 3], scale=0.8, size=(20, 2))

# 计算均值向量
mean_vector = np.mean(points, axis=0)

# 计算协方差矩阵
cov_matrix = np.cov(points, rowvar=False)

# 特征值和特征向量（协方差椭圆需要）
eigvals, eigvecs = np.linalg.eigh(cov_matrix)

# 排序特征值（从大到小）
order = eigvals.argsort()[::-1]
eigvals, eigvecs = eigvals[order], eigvecs[:, order]

# 协方差椭圆参数
angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))  # 主轴旋转角度
width, height = 2 * np.sqrt(eigvals)  # 1σ长度

# 绘制散点和均值向量点
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(points[:, 0], points[:, 1], color='blue', alpha=0.6, label='Pixel vectors $x_k$')
ax.scatter(mean_vector[0], mean_vector[1], color='red', s=100, label='Mean vector $\\mathbf{m}$')

# 添加协方差椭圆
ellipse = Ellipse(xy=mean_vector, width=width, height=height, angle=angle,
                  edgecolor='green', fc='None', lw=2, label='Covariance ellipse (1σ)')
ax.add_patch(ellipse)

# 坐标轴和标签
ax.set_xlabel('$x_1$ (Band 1 reflectance)', fontsize=12)
ax.set_ylabel('$x_2$ (Band 2 reflectance)', fontsize=12)
ax.set_title('Mean Vector and Covariance Ellipse in Spectral Space', fontsize=14, fontweight='bold')

# 坐标轴范围
ax.set_xlim(-1, 5)
ax.set_ylim(-1, 6)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend()

plt.show()
