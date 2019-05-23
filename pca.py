
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as plot
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# data = np.matrix([[149.5,162.5,162.7,162.2,156.5,156.1,172.0,173.2,159.5,157.7],
#               [69.5,77.0,78.5,87.5,74.5,74.5,76.5,81.5,74.5,79.0],
#               [38.5,55.5,50.8,65.5,49.0,45.5,51.0,59.5,43.5,53.5]])
# data = np.matrix(np.random.randint(0,50,size=[3,1000]))
# 解决画图时中文乱码的问题
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False


iris = load_iris()
data = np.mat(iris.data.T)


# 调用numpy实现pca
data_mean = np.mean(data,axis=1)
dataFinal = data-data_mean

data_cov = np.cov(dataFinal)
featValue,featVec = np.linalg.eig(data_cov)
print(featValue,featVec)

index = np.argsort(-featValue)

# 降到三维
contribution3 = (featValue[index[0]] + featValue[index[1]] + featValue[index[2]])/ sum(featValue) * 100
print(contribution3)
selectVec3 = np.matrix(featVec.T[index[:3]])
result3 = np.dot(selectVec3,dataFinal)
print(result3.T)
print("***********")
# 降到三维可视化
x3,y3,z3 = result3[0].tolist(),result3[1].tolist(),result3[2].tolist()
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(x3,y3,z3,c=iris.target)
plt.title("贡献率：" + str("%.2f" % contribution3) + "%")
plt.show()

# 降到二维
contribution2 = (featValue[index[0]] + featValue[index[1]])/ sum(featValue) * 100
print(contribution2)
selectVec2 = np.matrix(featVec.T[index[:2]])
result2 = np.dot(selectVec2,dataFinal)
print(result2.T)
print("***********")

# 降到二维可视化
x21,y21= result2[0,:50].tolist(),result2[1,:50].tolist()
x22,y22= result2[0,50:100].tolist(),result2[1,50:100].tolist()
x23,y23= result2[0,100:150].tolist(),result2[1,100:150].tolist()
fig2 = plt.figure()
plt.scatter(x21,y21,c='r')
plt.scatter(x22,y22,c='y')
plt.scatter(x23,y23,c='b')
plt.title("贡献率：" + str("%.2f" % contribution2) + "%")
plt.show()


# 降到一维
contribution1 = featValue[index[0]]/ sum(featValue) * 100
print(contribution1)
selectVec1 = np.matrix(featVec.T[index[:1]])
result = np.dot(selectVec1,dataFinal)
print(result.T)
print("***********")

# 直接调用pca
# pca = PCA(n_components=2)
# pca.fit(data.T)
# new = pca.fit_transform(data.T)
# print(new)


# 绘制原图像
# x,y,z = dataFinal[0],dataFinal[1],dataFinal[2]
# fig = plt.figure()
# ax = fig.add_subplot(111,projection='3d')
# ax.scatter(x,y,z)
# plt.show()


# 降到一维可视化
x1 = result[0].tolist()
fig1 = plt.figure()
y1 = []
for i in range(1,151):
    square = 0
    y1.append(square)
plt.scatter(x1,y1,c=iris.target)
plt.title("贡献率：" + str("%.2f" % contribution1 ) + "%")
plt.show()
