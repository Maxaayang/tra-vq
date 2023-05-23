import matplotlib.pyplot as plt

# 输入数据
x = [3, 6, 9, 12, 15]
y1 = [10, 8, 6, 4, 2]
y2 = [5, 4, 3, 2, 1]

# 绘制折线图
plt.plot(x, y1, label='Line 1')
plt.plot(x, y2, label='Line 2')

# 添加图例和标签
plt.legend()
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Line Chart')

# 显示图形
plt.show()
