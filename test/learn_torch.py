import torch

# 创建张量
torch.manual_seed(10)
a = torch.rand(2, 2, 5, 3)
# b = torch.rand([1, 1, 5, 3])
b = a[:,:,[0,1,3,4]]
print(b)
print(b.shape)
print("-----c ----------------")
c = torch.norm(b,dim=-1)
print(c)
print(c.shape)
d = torch.max(c,dim=1)[0]
print("------d ---------------")
print(d)
# print(d.shape)
# print(b)