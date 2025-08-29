import numpy as np
import matplotlib.pyplot as plt

# 模拟波长范围（单位：微米）
wavelength = np.linspace(0.3, 2.5, 300)

# 模拟太阳光谱（简化版高斯+指数衰减）
solar_spectrum = np.exp(-((wavelength - 0.5) ** 2) / (2 * 0.1 ** 2)) * 1.0 \
                 + 0.5 * np.exp(-(wavelength - 0.7) / 0.5)
solar_spectrum[solar_spectrum < 0] = 0

# 模拟校正后光谱（归一化到每个波段可比）
normalized_spectrum = solar_spectrum / np.max(solar_spectrum)

# 绘图
plt.figure(figsize=(8,5))
plt.plot(wavelength, solar_spectrum, label="Original Solar Spectrum", color="orange")
plt.plot(wavelength, normalized_spectrum, label="Normalized Spectrum", color="blue", linestyle="--")
plt.axvspan(0.4, 0.7, color="green", alpha=0.1, label="Visible Band")
plt.axvspan(0.7, 1.1, color="red", alpha=0.1, label="Near-Infrared Band")
plt.xlabel("Wavelength (μm)")
plt.ylabel("Relative Intensity")
plt.title("Effect of Normalization on Solar Spectrum")
plt.legend()
plt.grid(True)
plt.show()
