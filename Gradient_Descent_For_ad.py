import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义数据集大小
DATA_SIZE = 30

# 定义数据集
x1 = np.array([112, 203, 315, 423, 512, 614, 725, 838, 945, 1021,
               1102, 1210, 1325, 1423, 1512, 1618, 1725, 1832, 1940, 2015,
               2130, 2235, 2341, 2450, 2568, 2635, 2720, 2830, 2945, 3020])
x2 = np.array([15, 28, 37, 49, 58, 63, 77, 83, 91, 106,
               112, 125, 139, 145, 153, 162, 171, 185, 193, 204,
               215, 222, 234, 245, 252, 268, 276, 283, 297, 305])
x3 = np.array([7, 11, 18, 21, 28, 33, 38, 43, 49, 53,
               57, 63, 69, 73, 79, 82, 87, 93, 98, 102,
               109, 113, 118, 123, 128, 133, 138, 143, 147, 152])
y = np.array([162, 310, 455, 610, 755, 905, 1060, 1210, 1355, 1510,
              1660, 1810, 1965, 2110, 2260, 2415, 2565, 2710, 2865, 3010,
              3160, 3310, 3465, 3610, 3765, 3910, 4065, 4210, 4365, 4510])

# 学习率
yita = 0.0000001

# 参数初始化
xita = np.zeros(4)  # 这里定义了Θ0, Θ1, Θ2, Θ3

# 梯度计算函数
def grad(j):
    g = 0
    for i in range(DATA_SIZE):
        prediction = xita[0] + xita[1] * x1[i] + xita[2] * x2[i] + xita[3] * x3[i]
        error = prediction - y[i]
        if j == 0:
            g += error
        elif j == 1:
            g += error * x1[i]
        elif j == 2:
            g += error * x2[i]
        elif j == 3:
            g += error * x3[i]
    return g / DATA_SIZE

# 迭代次数
iterations = 100000
tolerance = 1e-7  # 提前停止条件的容忍度
previous_loss = 1e10  # 初始的前一轮损失值

# 梯度下降法
for iter in range(iterations):
    temp = np.zeros(4)  # 临时存储每次更新的参数值
    for j in range(4):
        temp[j] = xita[j] - yita * grad(j)
    xita = temp

    # 每10000次迭代输出一次参数和损失值
    if iter % 10000 == 0:
        loss = 0
        for i in range(DATA_SIZE):
            prediction = xita[0] + xita[1] * x1[i] + xita[2] * x2[i] + xita[3] * x3[i]
            error = prediction - y[i]
            loss += error ** 2
        loss /= 2 * DATA_SIZE
        print(f"Iteration {iter}: Loss = {loss}, xita0 = {xita[0]}, xita1 = {xita[1]}, xita2 = {xita[2]}, xita3 = {xita[3]}")

        # 提前停止条件
        if abs(previous_loss - loss) < tolerance:
            print(f"Early stopping at iteration {iter} due to small loss change.")
            break
        previous_loss = loss

# 预测
predicted_y = xita[0] + xita[1] * 700 + xita[2] * 80 + xita[3] * 10
print(f"Predicted y: {predicted_y}")
# 输出最终的参数值
print(f"Theta values: xita0 = {xita[0]}, xita1 = {xita[1]}, xita2 = {xita[2]}, xita3 = {xita[3]}")

# 绘图
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 绘制三维散点图
scatter = ax.scatter(x1, x2, x3, c=y, cmap='viridis', label='Data points')

# 绘制额外的点
extra_x1 = 700
extra_x2 = 80
extra_x3 = 10
ax.scatter(extra_x1, extra_x2, extra_x3, color='red', s=100, label='Extra point')

# 设置标题和坐标轴标签
ax.set_title('3D Scatter Plot of y vs x1, x2, x3')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('x3')

# 添加颜色条
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('y')

# 添加图例
ax.legend()

# 显示图形
plt.show()
