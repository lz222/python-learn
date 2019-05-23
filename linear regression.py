from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 加载数据集
boston = load_boston()
X = boston.data
Y = boston.target
df1 = pd.DataFrame(X, columns=boston.feature_names)
df2 = pd.DataFrame(Y)

# 归一化处理
mean = df1.mean(axis = 0)
std = df1.std(axis = 0)
df1 -= mean
df1 /= std



# 分割数据集，前404个为训练数据，后102个为验证数据
X_train_date = df1[:404]
X_verify_data = df1[404:]
Y_train = Y[:404]
Y_verify = Y[404:]

# 初始化参数和偏移量
theta = np.zeros((1,13))
b = 0

# 迭代次数、学习率、损失函数
iterations = 5000
alpha = 0.01
J = np.zeros( (iterations,1) )

# 梯度下降
for i in range(iterations):
    y_hat = np.dot(theta, X_train_date.T) + b
    c = y_hat - Y_train
    J[i] = (1 / (2 * 404)) * sum(sum((y_hat - Y_train) ** 2))
    temp0 = b - alpha * ((1 / 404) * sum(sum(y_hat - Y_train)))
    temp1 = theta - alpha * (1 / 404) * np.dot(y_hat - Y_train, X_train_date)
    b = temp0
    theta = temp1

# 预测验证数据的房价
Y_hat = np.dot(theta, X_verify_data.T) + b

# 画出预测值和真实值
plt.plot( range(len(Y_verify)), Y_verify, 'r', label='y_verify' )
plt.plot( range(len(Y_hat[0])), Y_hat[0], 'g--', label='y_predict' )
plt.title('sklearn: Linear Regression')
plt.legend()
plt.show()

# 损失函数
plt.plot( range(len(J)), J, 'b' )
plt.title('cost function')
plt.show()




