import matplotlib.pyplot as plt
import numpy as np

# 创建一个连续的信号
x = np.linspace(0, 2*np.pi, 500)
signal_high = np.sin(x) * 1.0        # 高方差信号
signal_low = np.sin(x) * 0.05        # 低方差信号

# 模拟量化（比如 8bit -> 256级，但这里只展示成粗量化以突出效果）
def quantize(signal, levels=16):
    s_min, s_max = signal.min(), signal.max()
    step = (s_max - s_min) / (levels - 1)
    return np.round((signal - s_min) / step) * step + s_min

signal_high_q = quantize(signal_high, levels=32)
signal_low_q = quantize(signal_low, levels=32)

# 绘图
fig, axes = plt.subplots(1, 2, figsize=(12,4))

# 高方差信号
axes[0].plot(x, signal_high, label="Original high variance", color="blue")
axes[0].plot(x, signal_high_q, label="Quantized", color="red", linestyle="--")
axes[0].set_title("High variance (PC1/PC2)\nQuantization not obvious")
axes[0].legend()

# 低方差信号
axes[1].plot(x, signal_low, label="Original low variance", color="blue")
axes[1].plot(x, signal_low_q, label="Quantized", color="red", linestyle="--")
axes[1].set_title("Low variance (PC5/PC6)\nQuantization dominates (noise)")
axes[1].legend()

plt.tight_layout()
plt.show()
