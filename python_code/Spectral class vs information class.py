import matplotlib.pyplot as plt
import numpy as np

# 模拟两类信息类：农作物（Crop）、沙地（Sand）
# 在光谱空间里它们可能出现重叠或分裂

np.random.seed(42)

# 光谱类数据 (模拟: 2个波段空间)
# 光谱类1 (部分Crop)
class1 = np.random.normal(loc=[2, 6], scale=0.6, size=(40, 2))
# 光谱类2 (另一部分Crop)
class2 = np.random.normal(loc=[5, 7], scale=0.6, size=(40, 2))
# 光谱类3 (Sand，但与Crop部分重叠)
class3 = np.random.normal(loc=[4, 4], scale=0.7, size=(40, 2))

# 绘图
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# 左图：光谱类 (Spectral Classes)
axes[0].scatter(class1[:, 0], class1[:, 1], c="blue", label="Spectral Class A")
axes[0].scatter(class2[:, 0], class2[:, 1], c="green", label="Spectral Class B")
axes[0].scatter(class3[:, 0], class3[:, 1], c="orange", label="Spectral Class C")
axes[0].set_title("Spectral Classes (Data Domain)")
axes[0].set_xlabel("Band 1 Reflectance")
axes[0].set_ylabel("Band 2 Reflectance")
axes[0].legend()

# 右图：信息类 (Information Classes)
# 假设 Class A 和 B 实际上都是农作物，Class C 是沙地
axes[1].scatter(np.vstack((class1, class2))[:, 0], np.vstack((class1, class2))[:, 1],
                c="green", label="Information Class: Crop")
axes[1].scatter(class3[:, 0], class3[:, 1], c="yellow", label="Information Class: Sand")
axes[1].set_title("Information Classes (Semantic Domain)")
axes[1].set_xlabel("Band 1 Reflectance")
axes[1].set_ylabel("Band 2 Reflectance")
axes[1].legend()

plt.tight_layout()
plt.show()
