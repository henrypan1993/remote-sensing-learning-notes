import numpy as np
import matplotlib.pyplot as plt

# 定义四种激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x >= 0, x, alpha * x)

# 生成输入范围
x = np.linspace(-6, 6, 400)

# 计算输出
y_sigmoid = sigmoid(x)
y_tanh = tanh(x)
y_relu = relu(x)
y_leaky_relu = leaky_relu(x)

# 绘图
plt.figure(figsize=(10, 8))

plt.plot(x, y_sigmoid, label="Sigmoid", color="blue")
plt.plot(x, y_tanh, label="tanh", color="green")
plt.plot(x, y_relu, label="ReLU", color="red")
plt.plot(x, y_leaky_relu, label="Leaky ReLU", color="orange")

plt.title("Common Activation Functions")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.axvline(0, color="black", linewidth=0.8, linestyle="--")

plt.show()
