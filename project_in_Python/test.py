import torch
import math
from sklearn.metrics.cluster import adjusted_mutual_info_score

a = torch.tensor([[5 / 17, 1 / 17, 2 / 17], [1 / 17, 4 / 17, 0 / 17], [0 / 17, 1 / 17, 3 / 17]])
sum_1 = torch.sum(a, dim = 1)
sum_2 = torch.sum(a, dim = 0)
print(sum_1)
print(sum_2)
res = 0
res_ = 0
for i in range(3):
    for j in range(3):
        if a[i][j] != 0:
            res += a[i][j].item() * torch.log2(a[i][j] / sum_1[i])
            res_ += a[i][j] * torch.log2(a[i][j])

print(res)
print(res_)

c1 = [1, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 1, 1, 3, 3, 3]
c2 = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]

c1 = [1, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 1, 1, 3, 3, 3]
c2 = [1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2]

ami = adjusted_mutual_info_score(c1, c2)
print('[INFO]AMI: ', ami)
