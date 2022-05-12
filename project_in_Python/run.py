from sklearn.preprocessing import MinMaxScaler
import functions
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import utlis
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score

from datetime import datetime

args = utlis.get_parse()
print(args)

start_time = datetime.now()
print('start time : ', start_time)

name = '../data/' + args.dataset

feat, label = utlis.read_file(name)

feat = torch.tensor(feat, dtype = float)
# print(label)
label = torch.tensor(label, dtype = float)
# scaler = MinMaxScaler()
# feat = scaler.fit_transform(feat)
# feat = torch.from_numpy(feat)
feat_std, feat_mean = torch.std_mean(feat, dim = 0, keepdim = False)
feat = (feat - feat_mean) / feat_std
# label = torch.squeeze(label, dim = 1)
print(feat, label)
print(type(feat))
# exit()
# l = functions.WBMS(X, 0.1, _lambda = 10)
l = functions.WBMS(feat, args.h, _lambda = args._lambda, tmax = args.tmax)
print(l[0])
print(l[1])
# plt.scatter(X[:, 0], X[:, 1])
# plt.scatter(l[0][:, 0], l[0][:, 1], c = 'black')
# plt.show()
res_label = functions.U2clus(l[0])
print(res_label)
res_label_tensor = torch.zeros(label.shape)
for i in res_label.keys():
    res_label_tensor[i] = res_label[i]
# res_label_tensor_paper = torch.tensor([1, 2, 1, 1, 3, 4, 5, 5, 3, 1, 2, 1, 1, 3, 4, 5, 5, 1, 1, 1, 1, 3, 4, 5, 5, 1, 1, 1, 1, 3, 4, 5, 5, 4, 6, 1, 1, 1, 1, 3, 4, 5, 5, 3, 1, 1, 1, 2, 3, 4, 5, 5, 4, 3, 1, 1, 1, 1, 3, 4, 5, 5, 4, 4, 1, 6, 6, 5, 6, 1, 1, 1, 1, 3, 4, 5, 5, 1, 1, 1, 1, 3, 4, 5, 5, 4, 4, 6, 6, 2, 1, 1, 1, 3, 4, 5, 5, 4, 4, 6, 6])
# res_label_tensor_paper = torch.tensor([1, 2, 1, 1, 3, 4, 5, 5, 3, 1, 2, 1, 1, 3, 6, 5, 5, 1, 1, 1, 1, 3, 4, 5, 5, 1, 1, 1, 1, 3, 6, 5, 5, 4, 7, 1, 1, 1, 1, 3, 6, 5, 5, 3, 1, 1, 1, 2, 3, 4, 5, 5, 4, 3, 1, 1, 1, 1, 3, 6, 5, 5, 4, 4, 1, 7, 7, 7, 7, 1, 1, 1, 1, 3, 6, 5, 5, 1, 1, 1, 1, 3, 6, 5, 5, 4, 4, 7, 7, 2, 1, 1, 1, 3, 6, 5, 5, 4, 4, 7, 7])
nmi = normalized_mutual_info_score(res_label_tensor, label)
print('NMI', nmi)
ari = adjusted_rand_score(label, res_label_tensor)
print('ARI', ari)
print('done')

end_time = datetime.now()
print('start time : ', start_time)
print('end time : ', end_time)
print('process time : ', end_time - start_time)