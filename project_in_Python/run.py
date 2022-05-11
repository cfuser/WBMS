from sklearn.preprocessing import MinMaxScaler
import functions
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import utlis
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score

X1 = torch.randn(100, 2)
# print(X1)
X2 = torch.randn(50, 2) + 5
X3 = torch.randn(50, 2) + 10
# print(X2)
X = torch.cat((X1, X2, X3), dim = 0)
# print(X)
for i in range(30):
    temp_x = torch.randn(200, 1)
    X = torch.cat((X, temp_x), dim = 1)
# print(X)

p = X.shape[1]
# print(p)

std, mean = torch.std_mean(X, dim = 0, keepdim = False)
# print(std, mean)
X = (X - mean) / std
fig = plt.figure("origin")
# ax = Axes3D(fig)
x = list(range(X.shape[0]))
x = [i - X.shape[0] / 2 for i in x]
y = list(range(X.shape[1]))
y = [j - X.shape[1] / 2 for j in y]
z = list(range(-1, 1 + 1))
# print(x)
# print(y)
# print(z)
# ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
# ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
# ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
# print(X.shape)
# for i in range(X.shape[0]):
#     ax.scatter(i, y, X[i, y], c='r')
# plt.show()
plt.scatter(X[:, 0], X[:, 1])
# plt.show()
new_fig = plt.figure("process")
# new_ax = Axes3D(new_fig)
# new_ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
# new_ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
# new_ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
name = '..\\data\\zoo.csv'
feat, label = utlis.read_file(name)

feat = torch.tensor(feat, dtype = float)
label = torch.tensor(label, dtype = float)
# scaler = MinMaxScaler()
# feat = scaler.fit_transform(feat)
# feat = torch.from_numpy(feat)
feat_std, feat_mean = torch.std_mean(feat, dim = 0, keepdim = False)
feat = (feat - feat_mean) / feat_std
label = torch.squeeze(label, dim = 1)
print(feat, label)
print(type(feat))
# exit()
# l = functions.WBMS(X, 0.1, _lambda = 10)
# l = functions.WBMS(feat, 0.08, _lambda = 20)
# print(l[0])
# print(l[1])
# # plt.scatter(X[:, 0], X[:, 1])
# # plt.scatter(l[0][:, 0], l[0][:, 1], c = 'black')
# # plt.show()
# res_label = functions.U2clus(l[0])
# print(res_label)
# res_label_tensor = torch.zeros(label.shape)
# for i in res_label.keys():
#     res_label_tensor[i] = res_label[i]
res_label_tensor_paper = torch.tensor([1, 2, 1, 1, 3, 4, 5, 5, 3, 1, 2, 1, 1, 3, 4, 5, 5, 1, 1, 1, 1, 3, 4, 5, 5, 1, 1, 1, 1, 3, 4, 5, 5, 4, 6, 1, 1, 1, 1, 3, 4, 5, 5, 3, 1, 1, 1, 2, 3, 4, 5, 5, 4, 3, 1, 1, 1, 1, 3, 4, 5, 5, 4, 4, 1, 6, 6, 5, 6, 1, 1, 1, 1, 3, 4, 5, 5, 1, 1, 1, 1, 3, 4, 5, 5, 4, 4, 6, 6, 2, 1, 1, 1, 3, 4, 5, 5, 4, 4, 6, 6])
res_label_tensor_paper = torch.tensor([1, 2, 1, 1, 3, 4, 5, 5, 3, 1, 2, 1, 1, 3, 6, 5, 5, 1, 1, 1, 1, 3, 4, 5, 5, 1, 1, 1, 1, 3, 6, 5, 5, 4, 7, 1, 1, 1, 1, 3, 6, 5, 5, 3, 1, 1, 1, 2, 3, 4, 5, 5, 4, 3, 1, 1, 1, 1, 3, 6, 5, 5, 4, 4, 1, 7, 7, 7, 7, 1, 1, 1, 1, 3, 6, 5, 5, 1, 1, 1, 1, 3, 6, 5, 5, 4, 4, 7, 7, 2, 1, 1, 1, 3, 6, 5, 5, 4, 4, 7, 7])
nmi = normalized_mutual_info_score(res_label_tensor_paper, label)
print('NMI', nmi)
ari = adjusted_rand_score(label, res_label_tensor_paper)
print('ARI', ari)
print('done')